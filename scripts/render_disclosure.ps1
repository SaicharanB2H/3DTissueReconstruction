$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$documentPath = Join-Path $projectDirectory 'Invention_Disclosure_Filled.docx'
$pdfPath = Join-Path $projectDirectory 'Invention_Disclosure_Filled.pdf'
$wordInstance = New-Object -ComObject Word.Application
$wordInstance.Visible = $false
$wordInstance.DisplayAlerts = 0
$wordInstance.Options.BackgroundSave = $false
$wordInstance.Options.Pagination = $false
$documentInstance = $null
try {
    Write-Output 'Opening disclosure for layout verification'
    $documentInstance = $wordInstance.Documents.Open($documentPath, $false, $true, $false)
    Write-Output 'Exporting PDF'
    $documentInstance.ExportAsFixedFormat($pdfPath, 17)
    Write-Output $pdfPath
} finally {
    if ($null -ne $documentInstance) { $documentInstance.Close(0) }
    $wordInstance.Quit(0)
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordInstance) | Out-Null
}
