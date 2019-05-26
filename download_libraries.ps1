# Download the OpenCV libraries from the cloud (stored in Dropbox and OneDrive)

# ffmpeg x64 dll
$destination = "lib/3rdParty/OpenCV/bin/opencv_ffmpeg410_64.dll"

$out_dir = Join-Path (Get-Location) "lib/3rdParty/OpenCV/bin"
if(!(Test-Path $out_dir))
{
	New-Item -ItemType directory -Path $out_dir
}

if(!([System.IO.File]::Exists( (Join-Path (Get-Location) $destination) )))
{		
	$source = "https://www.dropbox.com/s/gvkd4549wsjvn3u/opencv_ffmpeg410_64.dll?dl=1"
	Invoke-WebRequest $source -OutFile $destination
}

if(!([System.IO.File]::Exists( (Join-Path (Get-Location) $destination) )))
{
	$source = "https://onedrive.live.com/download?cid=2E2ADA578BFF6E6E&resid=2E2ADA578BFF6E6E%2153244&authkey=AEuLAF197Sy7S3M"
	Invoke-WebRequest $source -OutFile $destination
}

# Release x64 dll
$destination = "lib/3rdParty/OpenCV/x64/v141/bin/Release/opencv_world410.dll"

$out_dir = Join-Path (Get-Location) "lib/3rdParty/OpenCV/x64/v141/bin/Release"
if(!(Test-Path $out_dir))
{
	New-Item -ItemType directory -Path $out_dir
}

if(!([System.IO.File]::Exists( (Join-Path (Get-Location) $destination) )))
{
	$source = "https://www.dropbox.com/s/c81shi8br57xytv/opencv_world410.dll?dl=1"
	Invoke-WebRequest $source -OutFile $destination
}

if(!([System.IO.File]::Exists( (Join-Path (Get-Location) $destination) )))
{
	$source = "https://onedrive.live.com/download?cid=2E2ADA578BFF6E6E&resid=2E2ADA578BFF6E6E%2153246&authkey=AJkwseAGKqL3PpU"
	Invoke-WebRequest $source -OutFile $destination
}

# Debug x64 dll
$destination = "lib/3rdParty/OpenCV/x64/v141/bin/Debug/opencv_world410d.dll"

$out_dir = Join-Path (Get-Location) "lib/3rdParty/OpenCV/x64/v141/bin/Debug"
if(!(Test-Path $out_dir))
{
	New-Item -ItemType directory -Path $out_dir
}

if(!([System.IO.File]::Exists( (Join-Path (Get-Location) $destination) )))
{
	$source = "https://www.dropbox.com/s/8a4kmvpj5a09jdz/opencv_world410d.dll?dl=1"
	Invoke-WebRequest $source -OutFile $destination
}

if(!([System.IO.File]::Exists( (Join-Path (Get-Location) $destination) )))
{
	$source = "https://onedrive.live.com/download?cid=2E2ADA578BFF6E6E&resid=2E2ADA578BFF6E6E%2153247&authkey=AJE4pLhzjtkPDWs"
	Invoke-WebRequest $source -OutFile $destination
}