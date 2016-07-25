/**
* @Author: Chirag Raman <chirag>
* @Date:   2016-05-09T21:14:02-04:00
* @Email:  chirag.raman@gmail.com
* @Last modified by:   chirag
* @Last modified time: 2016-07-11T15:25:23-04:00
* @License: Copyright (C) 2016 Multicomp Lab. All rights reserved.
*/

#include <iostream>

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/avutil.h"
#include "libavutil/imgutils.h"
#include "libavdevice/avdevice.h"
}

/********
 * HELPERS
 *******/

static std::string AVStrError(int errnum) {
    char buf[128];
    av_strerror(errnum, buf, sizeof(buf));
    return std::string(buf);
}

/**
 * Copied from libav/cmdutils.c because unlike FFmpeg, Libav does not export
 * this function in the public API.
 */
std::string media_type_string(enum AVMediaType media_type) {
    switch (media_type) {
    case AVMEDIA_TYPE_VIDEO:      return "video";
    case AVMEDIA_TYPE_AUDIO:      return "audio";
    case AVMEDIA_TYPE_DATA:       return "data";
    case AVMEDIA_TYPE_SUBTITLE:   return "subtitle";
    case AVMEDIA_TYPE_ATTACHMENT: return "attachment";
    default:                      return "unknown";
    }
}

struct VideoDestintionAttributes {
    uint8_t *video_destination_data[4] = {NULL};
    int video_destination_linesize[4];
    int video_destination_bufsize;
};


/********
 * STATIC VARIABLES
 *******/

static AVCodecContext *video_decode_context = NULL;
static AVFrame *frame = NULL;
static int video_stream_index = -1;
static VideoDestintionAttributes video_dest_attr;
static FILE *video_destination_file = NULL;

/********
 * INITIALISATION
 *******/

void initialize() {
    av_register_all();
    avcodec_register_all();
    avformat_network_init();
    avdevice_register_all();
}

void init_format_context(AVFormatContext *&context) {
    context = avformat_alloc_context();
    if (!context) {
        av_log(0, AV_LOG_ERROR, "Cannot allocate input format (Out of memory?)\n");
        exit(1);
    }
    context->flags |= AVFMT_FLAG_NONBLOCK;
}

void init_input_format(AVInputFormat *&input_format) {
    input_format = av_find_input_format("v4l2");
    if (!input_format) {
        av_log(0, AV_LOG_ERROR, "Cannot find input format\n");
        exit(1);
    }
}

void init_options(AVDictionary *&options) {
    options = NULL;
    av_dict_set(&options, "framerate", "30", 0);
    av_dict_set(&options, "input_format", "h264", 0);
    av_dict_set(&options, "video_size", "1920x1080", 0);
}


/********
 * DEVICE INTERACTION
 *******/

void openCam(AVInputFormat *&input_format, AVDictionary *&options,
            AVFormatContext *&format_context, char const *device_name) {
    //Format context
    init_format_context(format_context);

    //Input Format
    init_input_format(input_format);

    //Options
    init_options(options);

    // check video source
    int open_result =
    avformat_open_input(&format_context, device_name, input_format, &options);

    if( open_result != 0) {
        std::cout << "\nOops, couldn't open video source:" << std::endl <<
         " error code  - " << open_result << std::endl <<
         " description - " << AVStrError(open_result) <<
         std::endl;
        exit(1);
    } else {
        av_log(0, AV_LOG_INFO, "Successfully opened camera!\n");
    }
}

void retrieve_stream_info(AVFormatContext *format_context) {
    if (avformat_find_stream_info(format_context, NULL) < 0) {
       av_log(0, AV_LOG_ERROR, "Could not find stream information\n");
       exit(1);
   }
}


/********
 * DECODING SETUP
 *******/

int open_codec_context(int *stream_index, AVFormatContext *format_context,
                       enum AVMediaType type, const char *device_name) {
    int ret = 0;
    AVStream *stream;
    AVCodecContext *decoder_context = NULL;
    AVCodec *decoder = NULL;
    AVDictionary *options = NULL;

    ret = av_find_best_stream(format_context, type, -1, -1, NULL, 0);
    if (ret < 0) {
        std::cerr << "Could not find " << media_type_string(type) <<
                  " stream from device " << device_name << std::endl;
    } else {
        *stream_index = ret;
        stream = format_context->streams[*stream_index];

        //Find decoder for the stream
        decoder_context = stream->codec;
        decoder = avcodec_find_decoder(decoder_context->codec_id);
        if (!decoder) {
            std::cerr << "Could not find the " << media_type_string(type) <<
                      " codec" << std::endl;
            ret = AVERROR(EINVAL);
        } else {
            //Init the decoders, with or without reference counting
            av_dict_set(&options, "refcounted_frames", "1", 0);
            ret = avcodec_open2(decoder_context, decoder, &options);
            if (ret < 0) {
                std::cerr << "Could not open the" <<  media_type_string(type)
                          << " codec" << std::endl;
            }
        }
    }
    return ret;
}

FILE* configure_video_destination_file(const char *filename) {
    FILE *dest_file = fopen(filename, "wb");
    if (!dest_file) {
        std::cerr <<  "Could not open destination file " << filename
                  << std::endl;
    }
    return dest_file;
}

int configure_video_buffer(AVCodecContext *video_decode_context,
                           VideoDestintionAttributes *dest_attr) {
    int ret = 0;

    // allocate image where the decoded image will be put
    ret = av_image_alloc(dest_attr->video_destination_data,
                         dest_attr->video_destination_linesize,
                         video_decode_context->width, video_decode_context->height,
                         video_decode_context->pix_fmt, 1);
    if (ret < 0) {
        std::cerr <<  "Could not allocate raw video buffer" << std::endl;
    } else {
        dest_attr->video_destination_bufsize = ret;
    }
    return ret;
}

int setup_frame(AVFrame *&frame) {
    int ret = 0;
    if (!(frame = av_frame_alloc())) {
        std::cerr << "Could not allocate frame" << std::endl;
        ret = -1;
    } else {
        // Set the fields of the given AVFrame to default values
        av_frame_unref(frame);
    }
    return ret;
}


/********
 * DECODE
 *******/
 int decode_packet(AVPacket packet,
                   int *got_frame,
                   int *video_frame_count,
                   int cached){
     int ret = 0;
     int decoded = packet.size;

     *got_frame = 0;

     if (packet.stream_index == video_stream_index) {
         //Decode video frame
         ret = avcodec_decode_video2(video_decode_context,
                                     frame,
                                     got_frame,
                                     &packet);
         if (ret < 0) {
             //FFmpeg users should use av_err2str
             char errbuf[128];
             av_strerror(ret, errbuf, sizeof(errbuf));
             std::cerr << "Error decoding video frame " << errbuf << std::endl;
             return ret;
         }

         if (*got_frame) {
             std::cout << "Video frame " << ( cached ? "(cached)" : "" )
                       << " n:" << (*video_frame_count)++
                       << " coded:" <<  frame->coded_picture_number
                       << " pts:" << frame->pts << std::endl;

             /*Copy decoded frame to destination buffer:
              *This is required since rawvideo expects non aligned data*/
             av_image_copy(video_dest_attr.video_destination_data,
                           video_dest_attr.video_destination_linesize,
                           (const uint8_t **)(frame->data),
                           frame->linesize,
                           video_decode_context->pix_fmt,
                           video_decode_context->width,
                           video_decode_context->height);

             //Write to rawvideo file
             fwrite(video_dest_attr.video_destination_data[0],
                    1,
                    video_dest_attr.video_destination_bufsize,
                    video_destination_file);
         }
     }

     if (*got_frame) {
         av_frame_unref(frame);
     }

     return decoded;
 }



/********
 * GRAB FRAMES
 *******/

int process_frames(AVFormatContext *context) {
    int ret = 0;
    int got_frame;
    int video_frame_count = 0;
    AVPacket packet;

    //Initialize packet, set data to NULL, let the demuxer fill it
    av_init_packet(&packet);
    packet.data = NULL;
    packet.size = 0;

    // read frames from the file
    for (;;) {
        AVPacket orig_packet;
        ret = av_read_frame(context, &packet);
        if (ret < 0) {
            if  (ret == AVERROR(EAGAIN)) {
                continue;
            } else {
                break;
            }
        }

        orig_packet = packet;
        do {
            ret = decode_packet(packet, &got_frame, &video_frame_count, 0);
            if (ret < 0) {
                break;
            }
            packet.data += ret;
            packet.size -= ret;
        } while (packet.size > 0);
        av_free_packet(&orig_packet);
    }

    //Flush cached frames
    packet.data = NULL;
    packet.size = 0;
    do {
        decode_packet(packet, &got_frame, &video_frame_count, 1);
    } while (got_frame);

    av_log(0, AV_LOG_INFO, "Demuxing succeeded\n");
    return ret;
}


/********
 * CLEANUP
 *******/

void cleanup(AVCodecContext *decode_context,
             AVFormatContext *format_context,
             FILE *dest_file,
             AVFrame *frame,
             VideoDestintionAttributes *attr) {
    avcodec_close(decode_context);
    avformat_close_input(&format_context);
    if(dest_file) { fclose(dest_file); }
    av_frame_free(&frame);
    av_free(attr->video_destination_data[0]);
}


/********
 * MAIN
 *******/

int main(int argc, const char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage : " << argv[0] << " device_path video_output_path"
                  << std::endl;
        exit(1);
    }

    //Declare local variables
    int ret = 0;
    char const *device_name = argv[1];
    const char *video_destination_filename = argv[2];

    AVInputFormat *input_format = NULL;
    AVDictionary *options = NULL;
    AVFormatContext *format_context = NULL;
    AVStream *video_stream = NULL;

    initialize();
    openCam(input_format, options, format_context, device_name);
    retrieve_stream_info(format_context);

    //Setup video decoding
    if (open_codec_context(&video_stream_index, format_context,
                           AVMEDIA_TYPE_VIDEO, device_name) >= 0) {
        video_stream = format_context->streams[video_stream_index];
        if (!video_stream) {
            std::cerr << "Could not find video stream in the input, aborting"
                      << std::endl;
        }

        video_decode_context = video_stream->codec;
        video_destination_file =
        configure_video_destination_file(video_destination_filename);

        int buffer_config = configure_video_buffer(video_decode_context,
                                                   &video_dest_attr);
        int frame_success = setup_frame(frame);

        //Abort if any step of the setup failed
        if (!video_stream || !video_destination_file || buffer_config < 0
            || frame_success != 0) {
            cleanup(video_decode_context,
                    format_context,
                    video_destination_file,
                    frame,
                    &video_dest_attr);
            exit(1);
        } else {
            av_log(0, AV_LOG_INFO, "Demuxing video from %s into %s\n",
                                    device_name, video_destination_filename);
            std::cout << "Play output video with the command:\n"
                      << "ffplay -f rawvideo -pixel_format "
                      << av_get_pix_fmt_name(video_decode_context->pix_fmt)
                      << " -video_size " << video_decode_context->width << "x"
                      << video_decode_context->height << " "
                      << video_destination_filename << std::endl <<std::endl;
        }
    }

    //Dump input information to stderr
    av_dump_format(format_context, 0, device_name, 0);
    std::cout<<std::endl;

    //Start processing frames
    ret = process_frames(format_context);

    return ret < 0;
}
