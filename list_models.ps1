$r = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags'
$r.models | Sort-Object name | ForEach-Object { '{0,-55} {1,8:N1} GB' -f $_.name, ($_.size / 1GB) }
