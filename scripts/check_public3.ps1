$ProgressPreference = 'SilentlyContinue'
$h = @{ 'User-Agent' = 'Mozilla/5.0 Chrome' }
$urls = @(
  'https://ai-inquiry-companion-lpqczxpzvtmerht4oddokv.streamlit.app/'
)
foreach ($u in $urls) {
  try {
    $r = Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 30 -Headers $h
    Write-Host ("OK  " + $u + "  STATUS=" + $r.StatusCode)
  } catch {
    Write-Host ("ERR " + $u + "  MSG=" + $_.Exception.Message)
    if ($_.Exception.Response) { Write-Host ("     ERR_STATUS=" + [int]$_.Exception.Response.StatusCode) }
  }
}