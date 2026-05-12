import os
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Setup Gemini ──
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash-lite')

app = Flask(__name__)
CORS(app)  # Allows your HTML (port 5500) to call this server (port 5000)


def build_prompt(prompt_type, user_data):
    """
    Build the correct Gemini prompt based on request type.
    Each type gets a specialized, detailed prompt for best results.
    """

    if prompt_type == "proposal":
        job_desc  = user_data.get('jobDesc', '')
        skill     = user_data.get('skill', 'Freelancer')
        exp       = user_data.get('exp', '1-2 years')
        urdu      = user_data.get('urdu', False)
        tones     = user_data.get('tones', ['Professional (Confident & Expert)'])

        urdu_note = "\n\nAfter each English proposal, add an Urdu translation labeled 'اردو ترجمہ:'." if urdu else ""

        return f"""You are an expert freelance proposal writer for Fiverr and Upwork.
Write {len(tones)} winning proposal(s) for the following job.

Freelancer Skill: {skill}
Experience Level: {exp}
Tone(s) needed: {', '.join(tones)}

Job Description:
\"\"\"
{job_desc}
\"\"\"

STRICT FORMAT — For each tone write:
[TONE NAME]
---
[Full proposal here]

EVERY proposal MUST follow this exact 3-part structure:

PART 1 - HUMAN OPENING (1-2 sentences):
Start with a warm, genuine greeting that references something SPECIFIC from the job description.
Good examples:
- "Hi! I just read your brief for [business name] and the [specific detail] really caught my eye."
- "Hello! Your project stood out to me immediately — the [specific requirement] is exactly the kind of challenge I enjoy."
Never start with: "I am writing to apply", "I saw your job posting", or "I am interested in this project."

PART 2 - BODY (4-6 sentences):
Explain what you will deliver, HOW you will do it, and WHY you are the right person.
Reference the client specific requirements by name. Adjust confidence to match the tone.

PART 3 - CALL TO ACTION (1 sentence):
End with a question that makes the client want to reply.
Example: "Shall we schedule a quick call to discuss your vision for [project name]?"

RULES:
- 90 to 130 words total per proposal
- Each tone must sound CLEARLY different in style and confidence
- Never use AI-sounding filler like "I am eager to leverage my expertise"
- Sound like a real human freelancer, warm and specific
- Be specific to THIS job description only{urdu_note}"""


    elif prompt_type == "gig":
        skill     = user_data.get('skill', '')
        platform  = user_data.get('platform', 'Fiverr')
        audience  = user_data.get('audience', 'small businesses and individuals')
        strengths = user_data.get('strengths', '')

        return f"""You are a top-rated {platform} seller who specializes in writing high-converting gig titles and descriptions.

Service: {skill}
Platform: {platform}
Target Audience: {audience}
Key Strengths: {strengths if strengths else 'not specified'}

Generate the following:

1. GIG TITLES (5 options)
Write 5 SEO-optimized gig titles. Each must:
- Be under 80 characters
- Include relevant keywords naturally
- Be specific and benefit-focused
- Be different from each other

2. GIG DESCRIPTION (200-250 words)
Structure it with these exact sections:
🎯 Opening Hook (1-2 sentences that grab attention)
✅ What You Get (bullet list of deliverables)
💪 Why Choose Me (2-3 differentiators)
📩 Call to Action (clear next step)

3. TAGS/KEYWORDS
List 5 relevant search tags for this gig.

Keep all text professional, energetic, and buyer-focused."""


    elif prompt_type == "fix":
        proposal  = user_data.get('proposal', '')
        job       = user_data.get('job', 'not specified')

        return f"""You are an expert editor for freelance proposals on Fiverr and Upwork.

Job context: {job}

Original proposal to fix:
\"\"\"
{proposal}
\"\"\"

Provide your response in TWO clearly labeled sections:

IMPROVED VERSION:
[Rewrite the proposal completely. Fix all grammar errors, improve sentence structure, make it more persuasive and professional. Keep it 80-130 words. The tone should be confident but not arrogant. Must end with a strong call to action.]

WHAT WAS FIXED:
[List exactly 4 specific improvements you made, as bullet points. Be specific, e.g. "Fixed grammar: 'I am having 3 year experience' → 'I have 3 years of experience'"]"""


    elif prompt_type == "score":
        proposal = user_data.get('proposal', '')

        return f"""You are a senior Upwork and Fiverr consultant who reviews proposals.

Proposal to score:
\"\"\"
{proposal}
\"\"\"

Respond in this EXACT format:

OVERALL SCORE: X/10

BREAKDOWN:
• Personalization: X/10 — [one sentence comment]
• Grammar & English: X/10 — [one sentence comment]
• Clarity of Offer: X/10 — [one sentence comment]
• Call to Action: X/10 — [one sentence comment]
• Persuasion Power: X/10 — [one sentence comment]

WHAT'S WORKING:
[2-3 specific things done well]

WHAT TO IMPROVE:
[3-4 specific, actionable suggestions with examples]

VERDICT:
[One honest sentence summarizing this proposal's chances of winning the job]"""


    elif prompt_type == "bio":
        platform   = user_data.get('platform', 'Instagram')
        name       = user_data.get('name', '')
        profession = user_data.get('profession', '')
        audience   = user_data.get('audience', 'potential clients')
        skills     = user_data.get('skills', '')
        tone       = user_data.get('tone', 'Professional')

        # Full instructions per platform — length + style + structure
        platform_rules = {
            'Instagram': (
                "LENGTH: Use the FULL 150 characters allowed. Do not write less than 120 characters.\n"
                "STRUCTURE: 3-4 short lines, each on a new line.\n"
                "Line 1: Who you are + what you do (with 1-2 emojis)\n"
                "Line 2: Who you help or your niche specialty\n"
                "Line 3: A result or value you deliver\n"
                "Line 4: CTA (e.g. DM me | Link in bio | Book a call)\n"
                "Use emojis naturally. Every line must add value."
            ),
            'TikTok': (
                "LENGTH: Maximum 80 characters. Write exactly 1-2 punchy lines.\n"
                "Make it hook the viewer in the first 3 words.\n"
                "Use 1-2 emojis. Be bold, fun, and memorable.\n"
                "Then add a very short CTA on the second line like: Follow for more."
            ),
            'Facebook': (
                "LENGTH: Use 200-255 characters. Write 2-3 warm sentences.\n"
                "Sentence 1: Who you are and what you do.\n"
                "Sentence 2: Who you help and what problem you solve.\n"
                "Sentence 3: A friendly CTA like: Send me a message to get started.\n"
                "Tone must be warm and community-oriented. Use 1-2 emojis."
            ),
            'LinkedIn': (
                "LENGTH: Write 100-150 words minimum. This is a full professional summary.\n"
                "STRUCTURE:\n"
                "Paragraph 1 (3-4 sentences): Hook + who you are + years of experience + what you specialize in.\n"
                "Paragraph 2 (3-4 sentences): Specific services you offer + types of clients you work with + measurable results if possible.\n"
                "Paragraph 3 (2-3 sentences): Your work approach/philosophy + a clear CTA.\n"
                "NO emojis. Formal tone. Sound like a senior professional."
            ),
            'Twitter/X': (
                "LENGTH: Use 130-160 characters. Do NOT write less than 100 characters.\n"
                "Write 2 lines:\n"
                "Line 1: Witty or bold statement about what you do (with niche).\n"
                "Line 2: Short CTA or personality note.\n"
                "Use max 2 emojis. Be punchy and memorable."
            ),
            'Fiverr Profile': (
                "LENGTH: Write 400-600 characters minimum. This is a full seller profile.\n"
                "STRUCTURE:\n"
                "Sentence 1-2: Strong opening hook about who you are and your top skill.\n"
                "Sentence 3-4: Specific services you offer and tools/platforms you use.\n"
                "Sentence 5-6: Why clients choose you — your unique value, delivery speed, revisions, etc.\n"
                "Sentence 7: Clear CTA like: Message me and let us get started today.\n"
                "No emojis. Professional but warm tone."
            ),
            'YouTube': (
                "LENGTH: Write 800-1000 characters for the About section.\n"
                "STRUCTURE:\n"
                "Line 1-2: Hook — what your channel is about and who it is for.\n"
                "Line 3-4: What viewers will learn or gain from subscribing.\n"
                "Line 5-6: Upload schedule and content style.\n"
                "Line 7: Subscribe CTA + business email note.\n"
                "Use emojis to separate sections. Energetic and exciting tone."
            ),
            'Telegram': (
                "LENGTH: Write 200-255 characters. This is a channel/group description.\n"
                "Line 1: What this channel is about in one bold statement.\n"
                "Line 2: Who should join and what they will get.\n"
                "Line 3: CTA like: Join now and stay updated.\n"
                "Use 2-3 emojis. Keep it punchy and clear."
            ),
            'GitHub': (
                "LENGTH: Write 160-250 characters for the GitHub profile bio.\n"
                "Keep it technical but human.\n"
                "Include: tech stack or specialty, what you build, and one fun fact or CTA.\n"
                "Example format: Full-stack dev | Python & React | Building open-source tools | Open to collabs\n"
                "Use minimal emojis (1-2 max)."
            ),
            'Upwork Profile': (
                "LENGTH: Write 200-280 words minimum. This is a detailed professional overview.\n"
                "STRUCTURE:\n"
                "Paragraph 1 (4-5 sentences): Strong intro — who you are, your specialty, years of experience, and what types of clients/projects you handle.\n"
                "Paragraph 2 (4-5 sentences): Specific services in detail — tools, platforms, deliverables, and any measurable results or achievements.\n"
                "Paragraph 3 (3-4 sentences): Your working style, communication approach, turnaround time, and a strong CTA.\n"
                "No emojis. Sound highly experienced and trustworthy."
            ),
        }

        rules = platform_rules.get(platform, platform_rules['Instagram'])

        return (
            f"You are an expert social media copywriter and personal branding specialist.\n\n"
            f"Write a {tone} bio for {platform} for this person:\n\n"
            f"Name: {name if name else 'not provided'}\n"
            f"Profession/Niche: {profession}\n"
            f"Target Audience: {audience}\n"
            f"Key Skills/Services: {skills if skills else 'not specified'}\n"
            f"Tone: {tone}\n\n"
            f"PLATFORM RULES FOR {platform.upper()}:\n"
            f"{rules}\n\n"
            f"CRITICAL RULES:\n"
            f"- You MUST meet the minimum length stated above. Short bios are WRONG.\n"
            f"- Sound like a real human being, not a robot\n"
            f"- Every sentence must add real value\n"
            f"- Do NOT add any explanation, commentary, or 'Why It Works' section\n"
            f"- Output ONLY the final bio text, nothing else\n"
            f"- Do NOT include labels like 'BIO:' or 'OUTPUT:' — just the bio itself"
        )

    else:
        return f"Respond to this freelance query: {user_data}"


@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data        = request.json
        prompt_type = data.get('type')   # "proposal", "gig", "fix", "score"
        user_data   = data.get('data', {})

        if not prompt_type:
            return jsonify({"message": "Missing 'type' field"}), 400

        prompt   = build_prompt(prompt_type, user_data)
        response = model.generate_content(prompt)

        return jsonify({"result": response.text})

    except Exception as e:
        print(f"Error: {e}")  # Shows in your terminal for debugging
        return jsonify({"message": str(e)}), 500


if __name__ == '__main__':
    print("✅ GigLikhari backend is running on http://127.0.0.1:5000")
    print("📌 Make sure your HTML file calls: http://127.0.0.1:5000/api/generate")
    app.run(port=5000, debug=True)
