# Electoral Roll OCR on EC2

This version runs the OCR pipeline manually on an existing Amazon Linux EC2 instance.
It does not use Lambda layers. Runtime tools such as Tesseract and Poppler are installed into a project-local conda-forge environment under `.runtime/`.

## GitHub Actions deploy

Add these repository secrets:

- `EC2_HOST`: public DNS or IP of the EC2 instance
- `EC2_USER`: SSH user, usually `ec2-user`
- `EC2_SSH_KEY`: private SSH key with access to the instance
- `EC2_PORT`: optional, defaults to `22`
- `APP_DIR`: optional, defaults to `/opt/eci-ocr`

Push to `main`, or run the `Deploy to EC2` workflow manually.

## Run on EC2

SSH into the instance:

```bash
ssh ec2-user@YOUR_EC2_HOST
cd /opt/eci-ocr
```

Copy a PDF into `inputs/`, then run:

```bash
scripts/run_pdf.sh inputs/sample.pdf
```

The JSON output is written to `outputs/sample_result.json`.

To insert into PostgreSQL, export `DATABASE_URL` before running:

```bash
export DATABASE_URL='postgresql://USER:PASSWORD@HOST:5432/DBNAME'
scripts/run_pdf.sh inputs/sample.pdf
```

To run without database insert:

```bash
scripts/run_pdf.sh inputs/sample.pdf outputs/result.json --skip-db
```

Local development:

```bash
make install
make test
make run PDF=inputs/sample.pdf
```
