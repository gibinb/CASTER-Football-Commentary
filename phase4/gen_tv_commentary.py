import groq, os, json
from dotenv import load_dotenv
load_dotenv()
client = groq.Groq(api_key=os.environ.get('GROQ_API_KEY'))

SYSTEM_PROMPT = """You are an elite live television football commentator broadcasting on Sky Sports (in the iconic style of Peter Drury).
Given the event, deliver ONE authentic, thrilling, poetic broadcast line (maximum 18 words) capturing the action on TV.
Never mention robot metrics, confidence, or 'contact confirmed'. No quotation marks."""

comms = json.load(open('outputs/ss_commentary.json', encoding='utf-8'))
for i in range(min(15, len(comms))):
    c = comms[i]
    ev = c['event']
    tm = c['time']
    prompt = f"Event: {ev} at match time {tm}. High energy broadcast action!"
    resp = client.chat.completions.create(
        model='openai/gpt-oss-120b',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=250,
        temperature=0.85
    )
    txt = resp.choices[0].message.content.strip().replace('"', '')
    c['commentary'] = txt
    print(f"[{i+1}/15] {tm} {ev} --> {txt}")

with open('outputs/ss_commentary.json', 'w', encoding='utf-8') as f:
    json.dump(comms, f, indent=2)
print("SUCCESS: Realistic TV commentary saved!")

