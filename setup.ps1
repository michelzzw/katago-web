<#
.SYNOPSIS
    KataGo Web 一键安装配置脚本 (Windows)
.DESCRIPTION
    自动下载 KataGo、最新权重模型，配置 Python 环境，启动 Web 服务器
#>

param(
    [string]$InstallDir = "C:\katago",
    [string]$Backend = "opencl"  # opencl, cuda, eigen (CPU)
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  KataGo Web 自动安装配置脚本" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ============== 1. 创建目录 ==============
Write-Host "[1/6] 创建安装目录..." -ForegroundColor Yellow
if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}
Write-Host "  安装目录: $InstallDir" -ForegroundColor Green

# ============== 2. 下载 KataGo ==============
Write-Host ""
Write-Host "[2/6] 下载 KataGo..." -ForegroundColor Yellow

$katagoExe = Join-Path $InstallDir "katago.exe"

if (Test-Path $katagoExe) {
    Write-Host "  KataGo 已存在，跳过下载" -ForegroundColor Green
} else {
    # 获取最新 release
    Write-Host "  正在查询最新版本..."
    $releaseUrl = "https://api.github.com/repos/lightvector/KataGo/releases/latest"
    try {
        $release = Invoke-RestMethod -Uri $releaseUrl -Headers @{"User-Agent"="KataGoWebInstaller"}
        $version = $release.tag_name
        Write-Host "  最新版本: $version"

        # 查找对应后端的 Windows 包
        $assetName = switch ($Backend) {
            "opencl" { "katago-*-opencl-windows-x64*" }
            "cuda"   { "katago-*-cuda*-windows-x64*" }
            "eigen"  { "katago-*-eigen*-windows-x64*" }
        }

        $asset = $release.assets | Where-Object { $_.name -like $assetName -and $_.name -like "*.zip" } | Select-Object -First 1
        if (-not $asset) {
            Write-Host "  警告: 未找到 $Backend 后端的 Windows 预编译包" -ForegroundColor Red
            Write-Host "  请手动从 https://github.com/lightvector/KataGo/releases 下载" -ForegroundColor Red
            Write-Host "  下载后将 katago.exe 放到 $InstallDir" -ForegroundColor Red
            Write-Host ""
            Write-Host "  可用资源:" -ForegroundColor Yellow
            $release.assets | ForEach-Object { Write-Host "    - $($_.name)" }
        } else {
            $zipPath = Join-Path $InstallDir "katago.zip"
            Write-Host "  下载: $($asset.name) ..."
            Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zipPath -UseBasicParsing
            
            Write-Host "  解压..."
            Expand-Archive -Path $zipPath -DestinationPath $InstallDir -Force
            
            # 查找解压后的 katago.exe
            $foundExe = Get-ChildItem -Path $InstallDir -Recurse -Filter "katago.exe" | Select-Object -First 1
            if ($foundExe -and $foundExe.FullName -ne $katagoExe) {
                # 把子文件夹内容移到安装目录
                $subDir = $foundExe.DirectoryName
                Get-ChildItem -Path $subDir | Move-Item -Destination $InstallDir -Force -ErrorAction SilentlyContinue
            }
            
            Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
            Write-Host "  KataGo 下载完成！" -ForegroundColor Green
        }
    } catch {
        Write-Host "  下载失败: $_" -ForegroundColor Red
        Write-Host "  请手动下载 KataGo: https://github.com/lightvector/KataGo/releases" -ForegroundColor Yellow
    }
}

# ============== 3. 下载最新权重 ==============
Write-Host ""
Write-Host "[3/6] 下载最新神经网络权重..." -ForegroundColor Yellow

$modelsDir = $InstallDir
$modelPattern = "kata1-*.bin.gz"
$existingModel = Get-ChildItem -Path $modelsDir -Filter $modelPattern -ErrorAction SilentlyContinue | Select-Object -First 1

if ($existingModel) {
    Write-Host "  已有模型: $($existingModel.Name)，跳过下载" -ForegroundColor Green
    $modelPath = $existingModel.FullName
} else {
    Write-Host "  正在从 KataGo 网站查询最新权重..."
    Write-Host ""
    Write-Host "  推荐权重 (按需选择):" -ForegroundColor Yellow
    Write-Host "    [1] kata1-b18c384nbt (推荐, ~70MB, 强度约职业级)" -ForegroundColor White
    Write-Host "    [2] kata1-b40c256    (较大, ~200MB, 更强)" -ForegroundColor White
    Write-Host "    [3] kata1-b6c96      (小型, ~15MB, 适合弱机器)" -ForegroundColor White
    Write-Host ""
    
    # 默认下载推荐的 b18c384nbt 权重
    $weightUrl = "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"
    $modelFileName = "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"
    $modelPath = Join-Path $modelsDir $modelFileName
    
    try {
        Write-Host "  下载 $modelFileName ..."
        Write-Host "  (文件约 70MB，请耐心等待)"
        Invoke-WebRequest -Uri $weightUrl -OutFile $modelPath -UseBasicParsing
        Write-Host "  权重下载完成！" -ForegroundColor Green
    } catch {
        Write-Host "  自动下载失败: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "  请手动下载权重文件:" -ForegroundColor Yellow
        Write-Host "  1. 访问 https://katagotraining.org/networks/" -ForegroundColor Yellow
        Write-Host "  2. 下载推荐权重 (b18c384nbt)" -ForegroundColor Yellow
        Write-Host "  3. 将 .bin.gz 文件放到 $modelsDir" -ForegroundColor Yellow
        $modelPath = Join-Path $modelsDir "MODEL_FILE_HERE.bin.gz"
    }
}

# ============== 4. 配置文件 ==============
Write-Host ""
Write-Host "[4/6] 配置文件..." -ForegroundColor Yellow

$logsDir = Join-Path $InstallDir "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
}
Write-Host "  配置完成" -ForegroundColor Green

# ============== 5. Python 环境 ==============
Write-Host ""
Write-Host "[5/6] 配置 Python 环境..." -ForegroundColor Yellow

$projectDir = Split-Path -Parent $PSScriptRoot
if (-not $projectDir) { $projectDir = $PSScriptRoot }
# 如果脚本直接在项目根目录运行
if (-not (Test-Path (Join-Path $projectDir "requirements.txt"))) {
    $projectDir = (Get-Location).Path
}

$reqFile = Join-Path $projectDir "requirements.txt"

if (Test-Path $reqFile) {
    Write-Host "  安装 Python 依赖..."
    try {
        pip install -r $reqFile 2>&1 | Out-Null
        Write-Host "  Python 依赖安装完成！" -ForegroundColor Green
    } catch {
        Write-Host "  pip 安装失败，尝试 python -m pip..." -ForegroundColor Yellow
        python -m pip install -r $reqFile
    }
} else {
    Write-Host "  未找到 requirements.txt，手动安装:" -ForegroundColor Yellow
    Write-Host "  pip install flask flask-socketio flask-cors eventlet" -ForegroundColor Yellow
}

# ============== 6. 生成启动脚本 ==============
Write-Host ""
Write-Host "[6/6] 生成启动脚本..." -ForegroundColor Yellow

$startScript = Join-Path $projectDir "start.ps1"
$startContent = @"
# KataGo Web 启动脚本
`$env:KATAGO_PATH = "$katagoExe"
`$env:KATAGO_MODEL = "$modelPath"
`$env:KATAGO_CONFIG = "$(Join-Path $projectDir 'config\default_gtp.cfg')"
`$env:PORT = "5000"
`$env:DEFAULT_MAX_VISITS = "3000"

Write-Host "启动 KataGo Web Server..." -ForegroundColor Cyan
Write-Host "  KataGo: `$env:KATAGO_PATH"
Write-Host "  Model:  `$env:KATAGO_MODEL"
Write-Host "  Port:   `$env:PORT"
Write-Host ""

# 获取本机 IP
`$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { `$_.InterfaceAlias -notlike "*Loopback*" -and `$_.IPAddress -notlike "169.*" } | Select-Object -First 1).IPAddress
Write-Host "  本机访问:   http://localhost:`$env:PORT" -ForegroundColor Green
Write-Host "  手机访问:   http://`${ip}:`$env:PORT" -ForegroundColor Green
Write-Host ""

Set-Location "$projectDir\server"
python app.py
"@

Set-Content -Path $startScript -Value $startContent -Encoding UTF8

# 同时生成 bat 启动脚本
$batScript = Join-Path $projectDir "start.bat"
$batContent = @"
@echo off
set KATAGO_PATH=$katagoExe
set KATAGO_MODEL=$modelPath
set KATAGO_CONFIG=$(Join-Path $projectDir 'config\default_gtp.cfg')
set PORT=5000
set DEFAULT_MAX_VISITS=3000

echo ============================================================
echo   KataGo Web Server
echo ============================================================
echo   KataGo: %KATAGO_PATH%
echo   Model:  %KATAGO_MODEL%
echo   Port:   %PORT%
echo ============================================================

cd /d "$projectDir\server"
python app.py
pause
"@

Set-Content -Path $batScript -Value $batContent -Encoding UTF8

Write-Host "  启动脚本已生成" -ForegroundColor Green

# ============== 完成 ==============
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  安装完成！" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  KataGo 路径:   $katagoExe" -ForegroundColor White
Write-Host "  模型路径:      $modelPath" -ForegroundColor White
Write-Host "  项目目录:      $projectDir" -ForegroundColor White
Write-Host ""
Write-Host "  启动方式:" -ForegroundColor Yellow
Write-Host "    方法1: 双击 start.bat" -ForegroundColor White
Write-Host "    方法2: PowerShell 中运行 .\start.ps1" -ForegroundColor White
Write-Host "    方法3: 手动运行 cd server && python app.py" -ForegroundColor White
Write-Host ""
Write-Host "  手机远程访问:" -ForegroundColor Yellow

try {
    $ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike "*Loopback*" -and $_.IPAddress -notlike "169.*" } | Select-Object -First 1).IPAddress
    Write-Host "    确保手机和电脑在同一 WiFi 网络" -ForegroundColor White
    Write-Host "    在手机浏览器访问: http://${ip}:5000" -ForegroundColor Green
} catch {
    Write-Host "    在手机浏览器访问: http://<你的电脑IP>:5000" -ForegroundColor White
}

Write-Host ""
Write-Host "  如果无法从手机访问，请:" -ForegroundColor Yellow
Write-Host "    1. 检查 Windows 防火墙是否放行 5000 端口" -ForegroundColor White
Write-Host "    2. 运行: netsh advfirewall firewall add rule name=`"KataGo Web`" dir=in action=allow protocol=tcp localport=5000" -ForegroundColor White
Write-Host ""
