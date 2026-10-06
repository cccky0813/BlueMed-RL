"""BlueMedical v2 Web 服务：Flask + SSE，把 LangGraph 每个节点的执行过程实时推给前端。

接口：
  GET  /                   页面
  POST /upload_image       上传图片 → 返回 image_id（文件名由服务端生成，不使用用户文件名）
  GET  /image/<image_id>   查看原图；GET /image/<image_id>/seg 查看分割图（只接受合法 image_id）
  POST /ask                {question, image_id?, thread_id} → SSE：每个 Agent 步骤一条 step 事件，最后 answer 事件
  GET  /history?thread_id= 读回该会话的历史问答和健康档案（服务重启后页面也能恢复对话）
运行：python app.py  → http://localhost:5002
"""

import json
import uuid

from flask import Flask, Response, abort, jsonify, render_template, request, send_file
from langchain_core.messages import AIMessage

from bluemed.graph import build_graph
from bluemed.memory.persistence import open_memory
from bluemed.memory.profile import load_profile
from bluemed.tools.image_tools import SEGMENT_DIR, resolve, save_upload

app = Flask(__name__)
checkpointer, store = open_memory()   # 短期 + 长期记忆都存进 data/memory.db，服务重启不丢
graph = build_graph(checkpointer, store)  # 全局一个实例，在多次请求间共享
USER_ID = "web_user"             # 演示用固定用户；真实系统应来自登录态
NODE_NAMES = {"parser": "① Question Parser Agent", "expert": "② Expert Agent", "tools": "🔧 工具",
              "generator": "③ Answer Generator", "reviewer": "③ Answer Reviewer", "remember": "💾 记忆"}


def describe_step(node: str, update: dict) -> str:
    """把一个节点的状态更新转成给人看的一行说明。"""
    if node == "parser":
        return f"类别：{update.get('category')}　改写：{update.get('question')}"
    if node == "expert":
        msg = update["messages"][-1]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            return "调用 " + "；".join(f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})" for c in msg.tool_calls)
        return msg.content
    if node == "tools":
        return "\n".join(m.content.splitlines()[0][:90] for m in update["messages"])
    if node == "generator":
        a = update["answer"]
        return f"生成回答：引用 {len(a.citations)} 条证据，证据不足={a.insufficient}"
    if node == "reviewer":
        r = update["review"]
        return f"{r.verdict}：{r.reason[:120]}"
    if node == "remember":
        return "本轮问答已写入会话记忆"
    return ""


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/history")
def history():
    thread_id = request.args.get("thread_id", "")
    state = graph.get_state({"configurable": {"thread_id": thread_id}}).values if thread_id else {}
    return jsonify(conversation=state.get("conversation", []), profile=load_profile(store, USER_ID))


@app.post("/upload_image")
def upload_image():
    file = request.files.get("image")
    if not file:
        return jsonify(error="没有收到图片"), 400
    try:
        image_id = save_upload(file.read())
    except Exception:
        return jsonify(error="无法识别的图片格式"), 400
    return jsonify(image_id=image_id, url=f"/image/{image_id}")


@app.get("/image/<image_id>")
@app.get("/image/<image_id>/<kind>")
def image(image_id, kind="raw"):
    path = resolve(image_id)                      # 只接受合法编号，防止通过 URL 读取任意文件
    if path is None:
        abort(404)
    if kind == "seg":
        path = SEGMENT_DIR / f"{image_id}_seg.png"
        if not path.exists():
            abort(404)
    return send_file(path)


@app.post("/ask")
def ask():
    body = request.get_json(force=True)
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify(error="问题为空"), 400
    image_id = body.get("image_id") if resolve(body.get("image_id") or "") else None
    config = {"configurable": {"thread_id": body.get("thread_id") or uuid.uuid4().hex, "user_id": USER_ID}}

    def stream():
        try:
            for step in graph.stream({"user_input": question, "image_id": image_id}, config, stream_mode="updates"):
                for node, update in step.items():
                    yield f"data: {json.dumps({'type': 'step', 'node': NODE_NAMES.get(node, node), 'text': describe_step(node, update)}, ensure_ascii=False)}\n\n"
            state = graph.get_state(config).values
            answer, evidence = state["answer"], state.get("evidence") or {}
            cited = [evidence[c].to_text() for c in answer.citations if c in evidence]
            seg = f"/image/{image_id}/seg" if image_id and (SEGMENT_DIR / f"{image_id}_seg.png").exists() else None
            payload = {"type": "answer", "answer": answer.answer, "safety": answer.safety_notice,
                       "evidence": cited, "profile": state.get("profile"), "segmented": seg}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as e:                    # 出错时也要通知前端，不能让页面一直等
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)[:200]}, ensure_ascii=False)}\n\n"

    return Response(stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=False, threaded=True)
