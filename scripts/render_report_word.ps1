param([string]$InputName='BCSE497J_Project_Report_23BAI1536.docx',[string]$OutputName='results/review/project_report/report_native.pdf',[switch]$UpdateFields)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$word=$null;$document=$null
try{
 $word=New-Object -ComObject Word.Application
 $word.Visible=$false;$word.DisplayAlerts=0
 $document=$word.Documents.Open((Join-Path $taskRoot $InputName),$false,(-not $UpdateFields),$false)
 Write-Output 'Opened document'
 if($UpdateFields){
   $document.Repaginate()
   foreach($toc in $document.TablesOfContents){$toc.Update()}
   $document.Fields.Update() | Out-Null
   foreach($toc in $document.TablesOfContents){
     $toc.Range.Font.Name='Times New Roman'
     $toc.Range.Font.Size=12
     $toc.Range.Font.Bold=0
     $toc.Range.ParagraphFormat.LineSpacingRule=0
     $toc.Range.ParagraphFormat.SpaceBefore=0
     $toc.Range.ParagraphFormat.SpaceAfter=2
   }
   $document.Repaginate()
   foreach($toc in $document.TablesOfContents){$toc.UpdatePageNumbers()}
   $document.Fields.Update() | Out-Null
   $document.Save()
 }
 $document.ExportAsFixedFormat((Join-Path $taskRoot $OutputName),17)
 Write-Output ('Exported '+$document.ComputeStatistics(2)+' pages')
}finally{if($null -ne $document){$document.Close(0)};if($null -ne $word){$word.Quit(0)}}
