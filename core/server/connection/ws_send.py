import json
import asyncio
from multiprocessing import Queue

from ..state import console
from ..schema import Result
from core.protocol import RecognitionMessage
from core.tools.asyncio_to_thread import to_thread
from .. import logger


def record_result(app, result: Result) -> None:
    """对单个识别结果做指标埋点，并在最终结果时写入持久化存储。

    该函数从发送循环中抽取为独立的纯同步逻辑，便于单元测试。
    """
    # 1. 任务计数（按来源打标签，支持 Prometheus 多维聚合）
    app.metrics.increment('caps_server_tasks_total', labels={'type': result.type})

    # 2. 识别耗时直方图（秒）
    process_time = result.time_complete - result.time_submit
    if process_time > 0:
        app.metrics.observe(
            'caps_server_recognition_duration_seconds', process_time
        )

    # 3. 实时率 RTF（Real-Time Factor = 处理耗时 / 音频时长）
    if result.duration > 0:
        app.metrics.set_gauge(
            'caps_server_rtf',
            process_time / result.duration,
            labels={'type': result.type},
        )

    # 4. 最终结果写入持久化存储
    if result.is_final:
        app.store.insert(
            task_id=result.task_id,
            source=result.type,
            text=result.text,
            text_accu=result.text_accu,
            duration=result.duration,
            tokens=result.tokens,
            timestamps=result.timestamps,
            time_start=result.time_start,
            time_submit=result.time_submit,
            time_complete=result.time_complete,
        )



async def ws_send(app):

    state = app.state
    queue_out = state.queue_out
    sockets = state.sockets

    logger.info("WebSocket 发送任务已启动")

    while True:
        try:
            # 更新输入队列积压指标（Windows 下 multiprocessing.Queue 支持 qsize）
            try:
                app.metrics.set_gauge(
                    'caps_server_queue_length',
                    float(state.queue_in.qsize()),
                )
            except (NotImplementedError, OSError):
                pass

            # 获取识别结果（从多进程队列）
            result: Result = await to_thread(queue_out.get)

            # 得到退出的通知
            if result is None:
                logger.info("收到退出通知，停止发送任务")
                return

            # 指标埋点 + 最终结果持久化
            record_result(app, result)

            # 1. 将内部 Result 转换为标准的协议消息对象
            msg = RecognitionMessage(
                task_id=result.task_id,
                is_final=result.is_final,
                duration=result.duration,
                time_start=result.time_start,
                time_submit=result.time_submit,
                time_complete=result.time_complete,
                text=result.text,
                text_accu=result.text_accu,
                tokens=result.tokens,
                timestamps=result.timestamps
            )

            # 获得 socket
            websocket = next(
                (ws for ws in sockets.values() if str(ws.id) == result.socket_id),
                None,
            )

            if not websocket:
                logger.warning(f"客户端 {result.socket_id} 不存在，跳过发送结果，任务ID: {result.task_id}")
                continue

            # 发送消息
            await websocket.send(msg.to_json())
            logger.debug(f"发送识别结果，任务ID: {result.task_id}, 文本长度: {len(result.text)}")

            if result.type == 'mic':
                logger.info(f"麦克风识别结果: {result.text}")
            elif result.type == 'file':
                console.print(f'    转录进度：{result.duration:.2f}s', end='\r')
                logger.debug(f"文件转录进度: {result.duration:.2f}s")
                if result.is_final:
                    console.print('\n    [green]转录完成')
                    logger.info(f"文件转录完成，任务ID: {result.task_id}, 总时长: {result.duration:.2f}s")

        except Exception as e:
            app.metrics.increment('caps_server_errors_total')
            logger.error(f"发送结果时发生错误: {e}", exc_info=True)
            print(e)


