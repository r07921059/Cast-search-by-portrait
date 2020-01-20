mkdir -p ./final_data

# Download dataset from Google drive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YqeqpSSS_QHvEzIcsjot-pp5jQDZrHk8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YqeqpSSS_QHvEzIcsjot-pp5jQDZrHk8" -O train.zip && rm -rf /tmp/cookies.txt

# Unzip the downloaded zip file
unzip ./train.zip -d ./final_data

# Remove the downloaded zip file
rm ./train.zip

# Download dataset from Google drive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1E63El9CKvm0YYntNp0M4uxIasgzzkXeD' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1E63El9CKvm0YYntNp0M4uxIasgzzkXeD" -O val.zip && rm -rf /tmp/cookies.txt

# Unzip the downloaded zip file
unzip ./val.zip -d ./final_data

# Remove the downloaded zip file
rm ./val.zip

# Download dataset from Google drive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dSzOCMHrMcSCcl0RSZQvhiFyYdVW9X0Q' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1dSzOCMHrMcSCcl0RSZQvhiFyYdVW9X0Q" -O test.zip && rm -rf /tmp/cookies.txt

# Unzip the downloaded zip file
unzip ./test.zip -d ./final_data

# Remove the downloaded zip file
rm ./test.zip

wget -O eval.py 'https://docs.google.com/uc?export=download&id=1Fd-HDBK459ZrfF9YQRIstEXw3L-28IKS'

wget -O train_GT.json 'https://docs.google.com/uc?export=download&id=1L9IUHuqB6g1zlj81r7p8FMh091IVWOUJ'

wget -O val_GT.json 'https://docs.google.com/uc?export=download&id=1Zr3B9e7Ra67nI9rFJ4JV-wXhJIUCEKHW'