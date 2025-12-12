import time
from functions import load_json, evaluate, print_full_report

chat_path = "sample_data/sample-chat-conversation-01.json"
context_path = "sample_data/sample_context_vectors-01.json"
start_time=time.perf_counter()

chat_data = load_json(chat_path)
context_data = load_json(context_path)

report = evaluate(chat_data, context_data)
print("----------------------")
print("OUTPUT FOR MAIN EXECUTION.")
print("----------------------")
print(f"COMPLETE SUMMARY: {report}")
print_full_report(report)
print(f"Time taken for main execution: {time.perf_counter()-start_time} seconds.")

# To test different inputs, change the values of these variables below
user_query = "What is the address of Dr Malpani clinic?"
required_context = "Address: Malpani Infertility Clinic, 505 Jamuna Sagar, near Colaba Bus Depot, Mumbai"
chatbot_response = "Dr. Malpani's clinic is in Mumbai on Shahid Bhagat Singh Road."

chat = {
        "chat_id": 1,
        "user_id": 1,
        "conversation_turns": [
            {
                "turn": 1,
                "sender_id": 1,
                "role": "User",
                "message": user_query,
                "created_at": "2025-12-11T10:00:00Z"
            },
            {
                "turn": 2,
                "sender_id": 2,
                "role": "AI/Chatbot",
                "message": chatbot_response,
                "created_at": "2025-12-11T10:00:10Z"
            }
        ]
    }

context = {
        "status": "success",
        "data": {
            "vector_data": [
                {
                    "id": 35674,
                    "source_url": "https://www.drmalpani.com/",
                    "text": required_context,
                    "tokens": 49,
                    "created_at": "2024-02-09T00:00:00Z"
                }
            ],
            "vectors_used": [35674]
        }
    }
# Individual Testing block 
# NOTE: Set 'individual_testing = True' to run this block with custom inputs for debugging or manual testing

individual_testing=True
if (individual_testing):
    start_time_ind=time.perf_counter()
    print("----------------------")
    print("OUTPUT FOR INDIVIDUAL TESTING.")
    print("----------------------")
    test_report=evaluate(chat, context)
    print(f"COMPLETE SUMMARY: {test_report}")
    print_full_report(test_report)
    print(f"Time taken for individual testing: {time.perf_counter()-start_time_ind} seconds.")