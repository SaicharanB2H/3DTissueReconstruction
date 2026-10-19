$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$documentPath = Join-Path $projectDirectory 'Invention_Disclosure_Final.docx'
$pdfPath = Join-Path $projectDirectory 'Invention_Disclosure_Final.pdf'
$statusPath = Join-Path $projectDirectory 'results\review\disclosure_final\word_render.json'
$logPath = Join-Path $projectDirectory 'results\review\disclosure_final\word_render.log'
$priorWordIds = @(Get-Process WINWORD -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id)
$wordInstance = $null
$documentInstance = $null
function Write-Stage([string]$stage) {
    $stage | Add-Content -LiteralPath $logPath -Encoding UTF8
    Write-Output $stage
}
try {
    Write-Stage 'Creating a hidden Word instance'
    $wordInstance = New-Object -ComObject Word.Application
    $wordInstance.Visible = $false
    $wordInstance.DisplayAlerts = 0
    $wordInstance.AutomationSecurity = 3
    Write-Stage 'Opening the corrected document read-only'
    $documentInstance = $wordInstance.Documents.Open($documentPath, $false, $true, $false)
    Write-Stage 'Exporting corrected document to PDF'
    $documentInstance.ExportAsFixedFormat($pdfPath, 17)
    Write-Stage 'PDF exported'
    @{status='success'; pdf=$pdfPath; priorWordIds=$priorWordIds} | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
} catch {
    Write-Stage $_.Exception.Message
    @{status='failed'; error=$_.Exception.Message; priorWordIds=$priorWordIds} | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
} finally {
    if ($null -ne $documentInstance) { $documentInstance.Close(0) }
    # This instance was created for rendering; close it after our document closes.
    if ($null -ne $wordInstance) {
        $wordInstance.Quit(0)
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordInstance) | Out-Null
    }
}
