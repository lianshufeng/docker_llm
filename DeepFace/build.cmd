docker build ./ -f Dockerfile --build-arg HTTP_PROXY=http://192.168.31.98:1080 --build-arg HTTPS_PROXY=http://192.168.31.98:1080 -t lianshufeng/llm:deepface
pause