This Docker container first builds a version of [TVM](https://github.com/apache/incubator-tvm) which is modified to print out the different TVM IRs at different stages of the build process.
It then runs a simple LSTM through TVM as an example.

# How it's useful
Though this is useful in viewing the different levels of TVM IR, it may be more useful as a guide, showing different places in TVM where IR is manipulated and lowered.
Please look in the Dockerfile to find the specific fork and commit of TVM that is being built.
Look through the commit history of this fork to see all of the places where IR is being printed.

# Running

```bash
git clone <this repo>
cd <this repo>
# Build Docker image from Dockerfile
docker build  -t 3la-ir-example  -f Dockerfile .
# Run Docker image (which runs the 3la-ir-example.py script)
docker run -it 3la-ir-example
```
