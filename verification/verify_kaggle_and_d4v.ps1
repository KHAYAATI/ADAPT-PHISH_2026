# Windows PowerShell — byte-level verification of the Kaggle copies and of D4v (run from the folder that contains Datasets\)
# Prerequisite: pip install kaggle ; put your kaggle.json token in %USERPROFILE%\.kaggle\ (Kaggle > Settings > API > Create token)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force verif | Out-Null
kaggle datasets download -d haozhang1579/crawling-2024 -p verif\crawling2024 --unzip
kaggle datasets download -d manishkc06/web-page-phishing-detection -p verif\manishkc06 --unzip
kaggle datasets download -d shashwatwork/web-page-phishing-detection-dataset -p verif\shashwatwork --unzip
"`n== SHA-256 of the downloaded copies =="
Get-ChildItem verif -Recurse -Filter *.csv | Get-FileHash -Algorithm SHA256 | Format-Table Hash, Path -AutoSize
"`n== SHA-256 of the files used by the notebook =="
Get-ChildItem Datasets -Filter *.csv | Get-FileHash -Algorithm SHA256 | Format-Table Hash, Path -AutoSize
"Compare: Crawling2024.csv <-> verif\crawling2024\dataset.csv ; Web_page_phishing_detection.csv <-> verif\manishkc06\*.csv ;"
"MendeleyPhishingWebsites.csv <-> verif\shashwatwork\dataset_phishing.csv. Equal hashes = byte-identical (write PASS in the audit)."
# D4v: download every file of https://data.mendeley.com/datasets/2ctjcnm9kn/1 by hand into verif\d4v\ (browser), then:
if (Test-Path verif\d4v) { Get-ChildItem verif\d4v -Recurse -File | Get-FileHash -Algorithm SHA256 | Format-Table Hash, Path -AutoSize }
