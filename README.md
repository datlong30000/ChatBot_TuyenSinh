# AI_lord

## Installation

Yêu cầu thư viện LangChain CLI

```bash
pip install -U langchain-cli
```

Ngoài ra hãy khởi tạo biến môi trường để có thể sử dụng API của các mô hình LLM


```bash
OPENAI_API_KEY = <API-KEY-OPENAI>
LANGCHAIN_API_KEY = <API-KEY-LANGSMITH> #Không bắt buộc
LANGCHAIN_TRACING_V2 = true #Không bắt buộc
LANGCHAIN_ENDPOINT = https://api.smith.langchain.com #Không bắt buộc
```

## Adding packages

```bash
# adding packages from 
# https://github.com/langchain-ai/langchain/tree/master/templates
langchain app add $PROJECT_NAME

# adding custom GitHub repo packages
langchain app add --repo $OWNER/$REPO
# or with whole git string (supports other git providers):
# langchain app add git+https://github.com/hwchase17/chain-of-verification

# with a custom api mount point (defaults to `/{package_name}`)
langchain app add $PROJECT_NAME --api_path=/my/custom/path/rag
```

Note: you remove packages by their api path

```bash
langchain app remove my/custom/path/rag
```


## Launch LangServe

```bash
langchain serve
```

## Running in Docker

This project folder includes a Dockerfile that allows you to easily build and host your LangServe app.

### Building the Image

To build the image, you simply:

```shell
docker build . -t my-langserve-app
```

If you tag your image with something other than `my-langserve-app`,
note it for use in the next step.

### Running the Image Locally

To run the image, you'll need to include any environment variables
necessary for your application.

In the below example, we inject the `OPENAI_API_KEY` environment
variable with the value set in my local environment
(`$OPENAI_API_KEY`)

We also expose port 8080 with the `-p 8080:8080` option.

```shell
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -p 8080:8080 my-langserve-app
```
