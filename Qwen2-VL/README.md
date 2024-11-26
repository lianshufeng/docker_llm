
- demo
````shell
curl --location 'http://127.0.0.1:8000/compatible-mode/v1' \
--header 'Content-Type: application/json' \
--data '{
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image":"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
                {"type": "text", "text": "图片里都有什么"}
            ]
        }
    ]
}'
````