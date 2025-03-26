import ollama
from colorama import init, Fore, Style

init()  # Initialize colorama for cross-platform colored terminal output

assistant_msg = 'You are an intelligent AI assistant that produces logical, honest, and helpful responses for human users.'
assistant_convo = [{'role': 'assistant', 'content': assistant_msg}]

optimize_msg = (
    "You are an expert AI prompt engineer specializing in rewriting user prompts to maximize the probability of generating "
    "the most relevant and useful response. Your task is to refine and optimize user prompts without changing their original "
    "intent, ensuring clarity, specificity, and effectiveness. \n\n"
    "Instructions:\n"
    "- You do not respond as an AI assistant.\n"
    "- You only return the improved version of the user prompt.\n"
    "- Do not include explanations, headers, or any additional text before or after the optimized prompt.\n"
    "- If the input contains a multi-turn conversation, analyze the entire context to generate the best possible refinement "
    "for the current user prompt within that conversation.\n"
    "- Maintain the original intent while improving structure, precision, and phrasing for optimal model response generation."
)  
optimized_convo = [{'role': 'assistant', 'content': optimize_msg}]

def generate_ollama_response(convo):
    response = ''
    stream = ollama.chat(
        model="llama3.1:8b",  # Replace with your Ollama model identifier
        messages=convo,
        stream=True,  # Enable streaming
    )

    for chunk in stream:
        content = chunk['message']['content']
        response += content
        # print(Fore.LIGHTWHITE_EX + content, end='', flush=True)
        
    convo.append({'role': 'assistant', 'content': response})
    return response

def main():
    global assistant_convo, optimized_convo

    while True:
        prompt = input(f'{Fore.CYAN}PROMPT: {Style.RESET_ALL}\n')
        if prompt == 'exit':
            break   
        optimized_prompt = generate_ollama_response(optimized_convo + [{'role': 'user', 'content': f'HUMAN PROMPT:\n {prompt}'}])
        print(f'\n\n{Fore.YELLOW}OPTIMIZED PROMPT: {Style.RESET_ALL}\n{optimized_prompt}\n\n')
        assistant_convo.append({'role': 'user', 'content': optimized_prompt})
        assistant_response = generate_ollama_response(assistant_convo)
        print(f'{Fore.GREEN}{Style.BRIGHT}ASSISTANT RESPONSE: {Style.RESET_ALL}\n{assistant_response}\n\n')
        assistant_convo.append({'role': 'assistant', 'content': assistant_response})
if __name__ == "__main__":
    main()
    
    



