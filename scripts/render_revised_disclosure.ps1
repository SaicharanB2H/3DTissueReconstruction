$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$documentPath = Join-Path $projectDirectory 'Invention_Disclosure_Revised.docx'
$pdfPath = Join-Path $projectDirectory 'Invention_Disclosure_Revised.pdf'
$statusPath = Join-Path $projectDirectory 'results\review\disclosure_revision\native_render_status.json'
$documentInstance = $null
$wordInstance = $null
try {
    # Reuse an accessible existing instance without closing the user's Word app.
    $wordInstance = [System.Runtime.InteropServices.Marshal]::GetActiveObject('Word.Application')
    $documentInstance = $wordInstance.Documents.Open($documentPath, $false, $true, $false)
    $documentInstance.Repaginate()
    $pages = $documentInstance.ComputeStatistics(2)
    $documentInstance.ExportAsFixedFormat($pdfPath, 17)
    @{ status = 'success'; pages = $pages; pdf = $pdfPath; userWordInstancePreserved = $true } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
    Write-Output "Exported revised disclosure: $pages pages"
} catch {
    @{ status = 'unavailable'; error = $_.Exception.Message; userWordInstancePreserved = $true } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
    Write-Output "Native rendering unavailable: $($_.Exception.Message)"
} finally {
    if ($null -ne $documentInstance) { $documentInstance.Close(0) }
    if ($null -ne $wordInstance) { [System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordInstance) | Out-Null }
}
