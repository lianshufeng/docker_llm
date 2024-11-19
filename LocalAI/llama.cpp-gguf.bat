@echo off

docker run --rm -v %CD%\%1:/models ghcr.io/ggerganov/llama.cpp:full --convert "/models" --outfile /models/model-f16.gguf --outtype f16
docker run --rm -v %CD%\%1:/models ghcr.io/ggerganov/llama.cpp:full --quantize "/models/model-f16.gguf" "/models/model-q4_k_m.gguf" "Q4_K_M"
pause