import http.server
import socketserver

PORT = 8888

class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        print("\n" + "🔥"*10 + " 成功捕获请求 " + "🔥"*10)
        print(f"👉 目标地址: {self.path}")
        print("👉 所有的 Headers (抄这些就对了):")
        print("-" * 30)
        # 打印所有 Header
        for header, value in self.headers.items():
            print(f"'{header}': '{value}'")
        print("-" * 30)
        
        # 任务完成，给它回个假数据防止卡死
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(b'{}')

# 允许地址重用，防止报错 Address already in use
socketserver.TCPServer.allow_reuse_address = True

with socketserver.TCPServer(("", PORT), ProxyHandler) as httpd:
    print(f"🕵️ 代理拦截器已启动，监听端口 {PORT}...")
    print("请在另一个窗口设置 HTTP_PROXY 环境变量...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止拦截")