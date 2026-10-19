$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$word=$null
$document=$null
try {
  $word=New-Object -ComObject Word.Application
  $word.Visible=$false
  $word.DisplayAlerts=0
  $document=$word.Documents.Open((Join-Path $taskRoot 'Invention_Disclosure_Word_Equations_Final.docx'),$false,$true,$false)
  Write-Output 'Opened equation copy read-only'
  $document.ExportAsFixedFormat((Join-Path $taskRoot 'results/review/disclosure_equations/native_word_equations.pdf'),17)
  Write-Output 'Exported native Word equation proof'
} finally {
  if($null -ne $document){$document.Close(0)}
  if($null -ne $word){$word.Quit(0)}
}
