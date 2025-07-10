#!/usr/bin/env python3
"""
快速测试脚本
"""

import requests
import json
import time
from typing import Dict, Any, List, Optional

# 全局配置
BASE_URL = "http://localhost:8000"
TIMEOUT = 600

def create_session(user_id: str = None) -> Optional[str]:
    """创建会话"""
    try:
        payload = {}
        if user_id:
            payload["user_id"] = user_id
        
        response = requests.post(f"{BASE_URL}/sessions", json=payload, timeout=TIMEOUT)
        if response.status_code == 200:
            session_id = response.json()["session_id"]
            print(f"[成功] 会话创建成功: {session_id[:8]}...")
            return session_id
        else:
            print(f"[错误] 会话创建失败: {response.status_code}")
            return None
    except Exception as e:
        print(f"[错误] 会话创建异常: {e}")
        return None

def send_query(query: str, session_id: str = None, use_context: bool = True, max_tokens: int = 512) -> Dict[str, Any]:
    """发送查询"""
    try:
        payload = {
            "query": query,
            "session_id": session_id,
            "use_context": use_context,
            "max_tokens": max_tokens
        }
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/query", json=payload, timeout=TIMEOUT)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            data["api_response_time"] = response_time
            return data
        else:
            return {
                "error": f"请求失败: {response.status_code}",
                "details": response.text
            }
    except Exception as e:
        return {
            "error": f"请求异常: {e}"
        }

def print_response(response: Dict[str, Any], show_details: bool = False):
    """打印响应结果"""
    if "error" in response:
        print(f"[错误] {response['error']}")
        if "details" in response:
            print(f"   详情: {response['details']}")
        return
    
    print(f"回答: {response.get('answer', 'N/A')}")
    
    if show_details:
        print(f"详细信息:")
        print(f"   响应时间: {response.get('response_time', 'N/A'):.2f}秒")
        print(f"   API响应时间: {response.get('api_response_time', 'N/A'):.2f}秒")
        print(f"   来源数量: {len(response.get('sources', []))}")
        print(f"   会话ID: {response.get('session_id', 'N/A')[:8]}...")
        
        # 安全检测结果
        toxicity = response.get('toxicity', {})
        print(f"   毒性检测: {'有毒' if toxicity.get('is_toxic', False) else '安全'}")
        
        hallucination = response.get('hallucination', {})
        print(f"   幻觉检测: {'有幻觉' if hallucination.get('is_hallucination', False) else '正常'}")

def single_test(query: str, show_details: bool = False):
    """单轮测试"""
    print(f"\n{'='*60}")
    print(f"单轮测试")
    print(f"{'='*60}")
    print(f"问题: {query}")
    
    response = send_query(query)
    print_response(response, show_details)
    
    return response

def multi_turn_test(questions: List[str], show_details: bool = False):
    """多轮测试"""
    print(f"\n{'='*60}")
    print(f"多轮测试 (共 {len(questions)} 轮)")
    print(f"{'='*60}")
    
    # 创建会话
    session_id = create_session()
    if not session_id:
        print("[错误] 无法创建会话，测试终止")
        return []
    
    responses = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- 第 {i} 轮 ---")
        print(f"问题: {question}")
        
        response = send_query(question, session_id)
        responses.append(response)
        
        print_response(response, show_details)

        if i < len(questions):
            time.sleep(1)
    
    return responses

def interactive_single_test():
    """交互式单轮测试"""
    print(f"\n{'='*60}")
    print(f"交互式单轮测试")
    print(f"输入 'quit' 或 'exit' 退出")
    print(f"{'='*60}")
    
    while True:
        try:
            query = input("\n请输入问题: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("测试结束")
                break
            
            if not query:
                print("[警告] 请输入有效问题")
                continue
            
            response = send_query(query)
            print_response(response, show_details=True)
            
        except KeyboardInterrupt:
            print("\n测试被中断")
            break
        except Exception as e:
            print(f"[错误] 发生异常: {e}")

def interactive_multi_turn_test():
    """交互式多轮测试"""
    print(f"\n{'='*60}")
    print(f"交互式多轮测试")
    print(f"输入 'quit' 或 'exit' 退出，'new' 创建新会话")
    print(f"{'='*60}")
    
    session_id = create_session()
    if not session_id:
        print("[错误] 无法创建会话")
        return
    
    turn_count = 0
    
    while True:
        try:
            query = input(f"\n第{turn_count + 1}轮 - 请输入问题: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("测试结束")
                break
            
            if query.lower() == 'new':
                session_id = create_session()
                if session_id:
                    turn_count = 0
                    print("[新建] 新会话已创建")
                continue
            
            if not query:
                print("[警告] 请输入有效问题")
                continue
            
            response = send_query(query, session_id)
            print_response(response, show_details=True)
            turn_count += 1
            
        except KeyboardInterrupt:
            print("\n测试被中断")
            break
        except Exception as e:
            print(f"[错误] 发生异常: {e}")

def test_enhanced_hallucination_detection():
    """测试幻觉检测"""
    print(f"\n{'='*60}")
    print(f"幻觉检测测试")
    print(f"{'='*60}")
    
    test_cases = [
        "什么药可以立即壮阳？",
        "前列腺炎的治疗方法有哪些？", 
        "包皮过长一定要手术吗？",
        "早泄能根治吗？"
    ]
    
    print("提前部署-需要API服务器运行")

    for i, query in enumerate(test_cases, 1):
        print(f"\n--- 测试 {i} ---")
        print(f"问题: {query}")
        
        try:
            response = send_query(query)
            
            if "error" in response:
                print(f"[错误] 请求失败: {response['error']}")
                continue
            
            print(f"回答: {response.get('answer', 'N/A')[:100]}...")
            
            hallucination = response.get('hallucination', {})
            print(f"幻觉检测: {hallucination.get('is_hallucination', 'N/A')}")
            print(f"置信度: {hallucination.get('confidence', 'N/A')}")
                
        except Exception as e:
            print(f"[错误] 测试失败: {e}")

def batch_test():
    """批量测试"""
    print(f"\n{'='*60}")
    print(f"批量测试")
    print(f"{'='*60}")
    
    test_questions = [
        "什么是高血压？",
        "糖尿病有哪些症状？",
        "感冒应该怎么治疗？",
        "如何预防心脏病？"
    ]
    
    print(f"测试 {len(test_questions)} 个问题")
    
    successful = 0
    total_time = 0
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- 测试 {i}/{len(test_questions)} ---")
        print(f"问题: {question}")
        
        start_time = time.time()
        response = send_query(question)
        test_time = time.time() - start_time
        total_time += test_time
        
        if "error" not in response:
            successful += 1
            print(f"[成功] {test_time:.2f}秒")
            print(f"回答: {response.get('answer', 'N/A')[:100]}...")
        else:
            print(f"[失败] {response['error']}")
    
    print(f"\n结果: {successful}/{len(test_questions)} 成功")
    print(f"平均时间: {total_time/len(test_questions):.2f}秒")

def conversation_context_test():
    """上下文对话测试"""
    print(f"\n{'='*60}")
    print(f"上下文对话测试")
    print(f"{'='*60}")
    
    conversation = [
        "我今年30岁，最近总是感觉头痛",
        "这种情况严重吗？",
        "我应该去医院检查什么？",
        "如果是高血压的话，应该怎么治疗？"
    ]
    
    print("测试上下文理解")
    responses = multi_turn_test(conversation, show_details=True)
    
    print(f"\n上下文分析:")
    keywords = ["头痛", "30岁", "高血压", "检查", "治疗"]
    
    for i, response in enumerate(responses, 1):
        if "error" not in response:
            answer = response.get('answer', '').lower()
            found = [kw for kw in keywords if kw in answer]
            print(f"第{i}轮关键词: {len(found)}/{len(keywords)}")

def performance_test(num_requests: int = 10):
    """性能测试"""
    print(f"\n{'='*60}")
    print(f"性能测试 ({num_requests} 个请求)")
    print(f"{'='*60}")
    
    test_query = "什么是高血压？"
    response_times = []
    successful = 0
    
    print(f"测试问题: {test_query}")
    
    start_total = time.time()
    
    for i in range(num_requests):
        print(f"请求 {i+1}/{num_requests}...", end=" ")
        
        response = send_query(test_query)
        
        if "error" not in response:
            response_time = response.get('api_response_time', 0)
            response_times.append(response_time)
            successful += 1
            print(f"[成功] {response_time:.2f}s")
        else:
            print(f"[失败]")
        
        time.sleep(0.5)
    
    total_time = time.time() - start_total
    
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        print(f"\n结果: {successful}/{num_requests} 成功")
        print(f"平均时间: {avg_time:.2f}秒")
        print(f"总时间: {total_time:.2f}秒")

def check_system_status():
    """检查系统状态"""
    print(f"\n{'='*60}")
    print(f"系统状态检查")
    print(f"{'='*60}")
    
    try:
        # 检查根端点
        response = requests.get(f"{BASE_URL}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("[成功] 系统运行正常")
            print(f"   版本: {data.get('version', 'N/A')}")
            print(f"   状态: {data.get('status', 'N/A')}")
        else:
            print(f"[错误] 系统状态异常: {response.status_code}")
    except Exception as e:
        print(f"[错误] 无法连接到系统: {e}")
        return False
    
    try:
        # 检查健康端点
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("[成功] 健康检查通过")
            print(f"   状态: {data.get('status', 'N/A')}")
        else:
            print(f"[错误] 健康检查失败: {response.status_code}")
    except Exception as e:
        print(f"[错误] 健康检查异常: {e}")
        return False
    
    return True

def main():
    """主菜单"""
    print("RAG系统快速测试工具")
    print("=" * 50)
    
    # 检查系统状态
    if not check_system_status():
        print("[错误] 系统不可用，请先启动RAG系统")
        return
    
    while True:
        print(f"\n测试选项:")
        print("1. 单轮测试 (输入单个问题)")
        print("2. 多轮测试 (预设问题序列)")
        print("3. 交互式单轮测试")
        print("4. 交互式多轮测试")
        print("5. 批量测试 (预设问题)")
        print("6. 上下文对话测试")
        print("7. 性能测试")
        print("8. 检查系统状态")
        print("9. 幻觉检测测试 (男科领域)")
        print("0. 退出")
        
        try:
            choice = input("\n请选择测试类型 (0-9): ").strip()
            
            if choice == "0":
                print("再见！")
                break
            elif choice == "1":
                query = input("请输入问题: ").strip()
                if query:
                    single_test(query, show_details=True)
                else:
                    print("[警告] 请输入有效问题")
            elif choice == "2":
                questions = []
                print("请输入问题序列 (输入空行结束):")
                while True:
                    q = input(f"问题 {len(questions)+1}: ").strip()
                    if not q:
                        break
                    questions.append(q)
                if questions:
                    multi_turn_test(questions, show_details=True)
                else:
                    print("[警告] 没有输入问题")
            elif choice == "3":
                interactive_single_test()
            elif choice == "4":
                interactive_multi_turn_test()
            elif choice == "5":
                batch_test()
            elif choice == "6":
                conversation_context_test()
            elif choice == "7":
                try:
                    num = int(input("请输入请求数量 (默认10): ").strip() or "10")
                    performance_test(num)
                except ValueError:
                    print("[警告] 请输入有效数字")
            elif choice == "8":
                check_system_status()
            elif choice == "9":
                test_enhanced_hallucination_detection()
            else:
                print("[警告] 无效选择，请输入 0-9")
                
        except KeyboardInterrupt:
            print("\n测试被中断")
            break
        except Exception as e:
            print(f"[错误] 发生异常: {e}")

if __name__ == "__main__":
    main() 