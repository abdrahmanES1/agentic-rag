# -*- coding: utf-8 -*-
"""
Centralized prompt templates for all pipeline LLM calls.
Single source of truth — changes here propagate everywhere.
"""

from typing import Dict

# ── Intent section templates (used in synthesis prompts) ─────────────────────

INTENT_SECTIONS_AR: Dict[str, str] = {
    "DOCUMENTS": "الوثائق المطلوبة:\nاذكر جميع الوثائق المطلوبة في نقاط واضحة. أضف [Source:...] بعد كل وثيقة.\n\n",
    "PROCEDURE": "الخطوات والإجراءات:\nاذكر خطوات الحصول على الخدمة مرتبة. أضف [Source:...] بعد كل خطوة.\n\n",
    "COST": "التكلفة والرسوم:\nاذكر المبالغ والرسوم بدقة فقط. أضف [Source:...] بعد الرقم.\n\n",
    "DEADLINE": "مدة الإنجاز:\nاذكر المدة الزمنية بدقة فقط. أضف [Source:...] بعد كل مدة.\n\n",
    "ELIGIBILITY": "شروط الأهلية:\nاذكر من يحق له التقديم والشروط المطلوبة. أضف [Source:...] بعد كل شرط.\n\n",
    "LEGAL": "العقوبات والجزاءات القانونية:\nاذكر العقوبات المنصوص عليها بدقة. أضف [Source:...] بعد كل عقوبة.\n\n",
    "COMPARISON": "المقارنة:\nقارن بين العنصرين: [الأول]: ... [الثاني]: ... [الفرق]: ...\nأضف [Source:...] بعد كل نقطة.\n\n",
}
INTENT_SECTIONS_FR: Dict[str, str] = {
    "DOCUMENTS": "Documents requis:\nListez tous les documents en points. Ajoutez [Source:...] après chaque document.\n\n",
    "PROCEDURE": "Étapes à suivre:\nListez les étapes dans l'ordre. Ajoutez [Source:...] après chaque étape.\n\n",
    "COST": "Frais et tarifs:\nIndiquez les montants exacts uniquement. Ajoutez [Source:...] après chaque montant.\n\n",
    "DEADLINE": "Délai de traitement:\nIndiquez le délai exact uniquement. Ajoutez [Source:...] après chaque délai.\n\n",
    "ELIGIBILITY": "Conditions d'éligibilité:\nIndiquez qui peut postuler et les conditions. Ajoutez [Source:...] après chaque condition.\n\n",
    "LEGAL": "Sanctions et pénalités:\nIndiquez les sanctions avec précision. Ajoutez [Source:...] après chaque sanction.\n\n",
    "COMPARISON": "Comparaison:\n[Premier]: ... [Second]: ... [Différence]: ...\nAjoutez [Source:...] après chaque point.\n\n",
}
INTENT_SECTIONS_DA: Dict[str, str] = {
    "DOCUMENTS": "الوثائق المحتاجين ليها:\nذكر الوثائق المطلوبة واحدة واحدة. زيد [Source:...] بعد كل وثيقة.\n\n",
    "PROCEDURE": "الخطوات:\nذكر الخطوات بالترتيب. زيد [Source:...] بعد كل خطوة.\n\n",
    "COST": "الثمن والرسوم:\nذكر المبلغ بالضبط. زيد [Source:...] بعد كل رقم.\n\n",
    "DEADLINE": "المدة:\nذكر الوقت الحقيقي بالضبط. زيد [Source:...] بعد كل مدة.\n\n",
    "ELIGIBILITY": "الشروط:\nذكر من يقدر يطلب وعلاش. زيد [Source:...] بعد كل شرط.\n\n",
    "LEGAL": "العقوبات:\nذكر العقوبات بالضبط. زيد [Source:...] بعد كل عقوبة.\n\n",
    "COMPARISON": "المقارنة:\n[الأول]: ... [الثاني]: ... [الفرق]: ...\nزيد [Source:...] بعد كل نقطة.\n\n",
}


def get_intent_sections(language: str) -> Dict[str, str]:
    if language == "Darija":
        return INTENT_SECTIONS_DA
    if language in ("arabic_msa", "Arabizi"):
        return INTENT_SECTIONS_AR
    return INTENT_SECTIONS_FR


# ── Generation prompts ────────────────────────────────────────────────────────

def direct_generation_prompt(question: str, context: str, language: str) -> str:
    if language == "Darija":
        return (
            "أنت مساعد إداري رسمي متخصص في الخدمات العامة المغربية.\n"
            "مهمتك: جاوب على سؤال المواطن بناءً على الوثائق الرسمية فقط.\n\n"
            "🔴 القواعد الإلزامية:\n"
            "1. جاوب بالدارجة المغربية فقط.\n"
            "2. استعمل غير المعلومات لي كاينة فالوثائق — ماتزيدش حاجة من عندك.\n"
            "3. الوثائق يمكن تكون بالعربية أو الفرنسية — الزوج مصدر رسمي — اقراهم كلهم.\n"
            "4. إلا لقيتي وثيقة بالفرنسية تجاوب السؤال: استعملها وترجمها للدارجة.\n"
            "5. إلا ماكانتش المعلومة فالوثائق: قول 'هاد المعلومة ماكاينةش فالوثائق المتاحة.'\n"
            "6. زيد [Source: filename.pdf] بعد كل معلومة مباشرة.\n\n"
            f"الوثائق الرسمية:\n{context}\n\n"
            f"سؤال المواطن: {question}\n\n"
            "الجواب بالدارجة:\n"
        )
    if language == "arabic_msa":
        return (
            "أنت مساعد إداري رسمي متخصص في الخدمات العامة المغربية.\n"
            "مهمتك: الإجابة على سؤال المواطن بناءً على الوثائق الرسمية المقدمة فقط.\n\n"
            "🔴 القواعد الإلزامية:\n"
            "1. اكتب إجابتك بالعربية الفصحى فقط.\n"
            "2. لا تخترع أي معلومة غير موجودة في الوثائق — هذا محظور تماماً.\n"
            "3. الوثائق قد تكون بالعربية أو الفرنسية — كلاهما مصدر رسمي صالح — اقرأ الجميع.\n"
            "4. إذا وجدت وثيقة بالفرنسية تجيب على السؤال: استخدمها وترجمها للعربية.\n"
            "5. إذا لم تجد المعلومة في الوثائق: قل 'هذه المعلومة غير متوفرة في الوثائق المتاحة.'\n"
            "6. أضف [Source: filename.pdf] بعد كل معلومة مباشرة.\n\n"
            f"الوثائق الرسمية:\n{context}\n\n"
            f"سؤال المواطن: {question}\n\n"
            "الإجابة:\n"
        )
    if language == "Arabizi":
        return (
            "You are an official administrative assistant specialized in Moroccan public services.\n"
            "Mission: answer the citizen's question using Moroccan Darija in Arabizi with clear words.\n\n"
            "🔴 Mandatory rules:\n"
            "1. Write your answer ONLY in Moroccan Darija in Arabizi.\n"
            "2. Do NOT invent ANY information absent from the documents — strictly forbidden.\n"
            "3. Documents may be in Arabic or French — both are valid official sources — read all.\n"
            "4. If you find a French document answering the question: use it and translate to Arabizi.\n"
            "5. If information is missing: say 'had lma3luma makaynach f lwatha2eq lmota7.'\n"
            "6. Add [Source: filename.pdf] immediately after each fact.\n\n"
            f"Official documents:\n{context}\n\n"
            f"Citizen question: {question}\n\n"
            "Answer in Arabizi:\n"
        )
    # French
    return (
        "Vous êtes un assistant administratif officiel spécialisé dans les services publics marocains.\n"
        "Votre mission: répondre à la question du citoyen en vous basant UNIQUEMENT sur les documents officiels fournis.\n\n"
        "🔴 Règles impératives:\n"
        "1. Rédigez votre réponse UNIQUEMENT en français.\n"
        "2. N'inventez AUCUNE information absente des documents — strictement interdit.\n"
        "3. Les documents peuvent être en arabe ou en français — les deux sont des sources officielles valides.\n"
        "4. Si un document en arabe répond à la question: utilisez-le et traduisez son contenu en français.\n"
        "5. Si l'information est absente: indiquez 'Cette information n'est pas disponible dans les documents fournis.'\n"
        "6. Ajoutez [Source: filename.pdf] immédiatement après chaque information.\n\n"
        f"Documents officiels:\n{context}\n\n"
        f"Question du citoyen: {question}\n\n"
        "Réponse:\n"
    )


def synthesis_prompt(question: str, facts_context: str, section_instructions: str, language: str) -> str:
    if language == "Darija":
        return (
            "أنت مساعد إداري رسمي. مهمتك: اجمع الأجوبة الجزئية في جواب واحد شامل ومنظم.\n\n"
            "🔴 القواعد الإلزامية:\n"
            "1. اكتب الجواب النهائي بالدارجة المغربية فقط.\n"
            "2. استعمل غير المعلومات لي كاينة فالأجوبة الجزئية — ماتزيدش حاجة من عندك.\n"
            "3. حافظ على [Source: filename.pdf] كما هي في كل مكان.\n"
            "4. ماتكررش نفس المعلومة في أقسام مختلفة.\n"
            "5. رتب الأقسام بالترتيب الطبيعي: الوثائق → الخطوات → التكلفة → المدة.\n\n"
            f"السؤال الأصلي: {question}\n\n"
            f"الأجوبة الجزئية:\n{facts_context}\n"
            f"{section_instructions}"
            "الجواب النهائي الشامل بالدارجة:\n"
        )
    if language in ("arabic_msa", "Arabizi"):
        return (
            "أنت مساعد إداري رسمي. مهمتك: اجمع الإجابات الفرعية في إجابة واحدة شاملة ومنظمة.\n\n"
            "🔴 القواعد الإلزامية:\n"
            "1. اكتب الإجابة النهائية بالعربية الفصحى فقط.\n"
            "2. استخدم فقط المعلومات الموجودة في الإجابات الفرعية — لا تخترع شيئاً.\n"
            "3. حافظ على [Source: filename.pdf] كما هي في كل مكان.\n"
            "4. لا تكرر نفس المعلومة في أقسام مختلفة.\n"
            "5. رتب الأقسام بالترتيب الطبيعي: الوثائق → الإجراءات → التكلفة → المدة.\n\n"
            f"السؤال الأصلي: {question}\n\n"
            f"الإجابات الفرعية:\n{facts_context}\n"
            f"{section_instructions}"
            "الإجابة النهائية الشاملة:\n"
        )
    return (
        "Vous êtes un assistant administratif officiel. Votre mission: assembler les réponses partielles en une réponse complète et structurée.\n\n"
        "🔴 Règles impératives:\n"
        "1. Rédigez la réponse finale UNIQUEMENT en français.\n"
        "2. Utilisez UNIQUEMENT les informations présentes dans les réponses partielles — n'inventez rien.\n"
        "3. Conservez exactement les balises [Source: filename.pdf] à leur place.\n"
        "4. Ne répétez pas la même information dans des sections différentes.\n"
        "5. Organisez dans l'ordre naturel: documents → étapes → coût → délai.\n\n"
        f"Question originale: {question}\n\n"
        f"Réponses partielles:\n{facts_context}\n"
        f"{section_instructions}"
        "Réponse finale complète:\n"
    )


def intermediate_generation_prompt(
    sub_question: str,
    intent: str,
    chunk_context: str,
    prior_context: str,
    language: str,
) -> str:
    intent_instr = {
        "arabic_msa": {
            "DOCUMENTS": "اذكر الوثائق المطلوبة فقط في نقاط.",
            "PROCEDURE": "اذكر الخطوات فقط مرتبة.",
            "COST": "اذكر المبالغ والرسوم فقط بأرقام.",
            "DEADLINE": "اذكر المدة الزمنية فقط.",
            "ELIGIBILITY": "اذكر شروط الأهلية فقط.",
            "LEGAL": "اذكر العقوبات القانونية فقط.",
            "COMPARISON": "قارن بين العنصرين فقط.",
        },
        "Darija": {
            "DOCUMENTS": "ذكر الوثائق المحتاجين ليها واحدة واحدة.",
            "PROCEDURE": "ذكر الخطوات بالترتيب.",
            "COST": "ذكر الثمن والرسوم بالأرقام.",
            "DEADLINE": "ذكر المدة بالضبط.",
            "ELIGIBILITY": "ذكر الشروط ديال التقديم.",
            "LEGAL": "ذكر العقوبات.",
            "COMPARISON": "قارن بين الزوج.",
        },
        "french": {
            "DOCUMENTS": "Listez uniquement les documents requis.",
            "PROCEDURE": "Listez uniquement les étapes dans l'ordre.",
            "COST": "Indiquez uniquement les montants exacts.",
            "DEADLINE": "Indiquez uniquement le délai exact.",
            "ELIGIBILITY": "Indiquez uniquement les conditions d'éligibilité.",
            "LEGAL": "Indiquez uniquement les sanctions prévues.",
            "COMPARISON": "Comparez uniquement les deux éléments.",
        },
        "Arabizi": {
            "DOCUMENTS": "List only the required documents in points.",
            "PROCEDURE": "List only the steps in order.",
            "COST": "State only the exact amounts and fees.",
            "DEADLINE": "State only the exact deadline/duration.",
            "ELIGIBILITY": "State only the eligibility conditions.",
            "LEGAL": "State only the legal penalties.",
            "COMPARISON": "Compare only the two elements.",
        },
    }

    lang_key = language if language in intent_instr else "french"
    instr = intent_instr[lang_key].get(intent, "أجب على السؤال بإيجاز." if lang_key in ("arabic_msa", "Darija") else "Répondez brièvement.")

    if language == "Darija":
        return (
            "أنت مساعد إداري متخصص في الخدمات العامة المغربية. جاوب بالدارجة المغربية.\n"
            "استعمل غير المعلومات لي كاينة فالوثيقة. ماتزيدش حاجة من عندك.\n"
            f"{prior_context}"
            f"الوثيقة:\n{chunk_context}\n\n"
            f"السؤال: {sub_question}\n"
            f"التعليمات: {instr}\n"
            "زيد [Source: filename.pdf] بعد كل معلومة.\n"
            "الجواب:"
        )
    if language == "arabic_msa":
        return (
            "أنت مساعد إداري. أجب بإيجاز باستخدام الوثيقة فقط.\n"
            f"{prior_context}"
            f"الوثيقة:\n{chunk_context}\n\n"
            f"السؤال: {sub_question}\n"
            f"التعليمات: {instr}\n"
            "أضف [Source: filename.pdf] بعد كل معلومة.\n"
            "الإجابة المختصرة:"
        )
    if language == "Arabizi":
        return (
            "You are an administrative assistant. Answer briefly in Moroccan Darija in Arabizi.\n"
            "Use only information from the document.\n"
            f"{prior_context}"
            f"Document:\n{chunk_context}\n\n"
            f"Question: {sub_question}\n"
            f"Instruction: {instr}\n"
            "Add [Source: filename.pdf] after each fact.\n"
            "Short answer in Arabizi:"
        )
    return (
        "Vous êtes un assistant administratif. Répondez brièvement à partir du document uniquement.\n"
        f"{prior_context}"
        f"Document:\n{chunk_context}\n\n"
        f"Question: {sub_question}\n"
        f"Instruction: {instr}\n"
        "Ajoutez [Source: filename.pdf] après chaque information.\n"
        "Réponse courte:"
    )


# ── Classification prompts ────────────────────────────────────────────────────

CLASSIFIER_SYSTEM_AR = (
    "أنت مصنف أسئلة لنظام خدمات عامة مغربية.\n"
    "السؤال قد يحتوي على نوايا متعددة في آن واحد.\n\n"
    "تعريف النوايا:\n"
    "- DOCUMENTS: يسأل عن وثائق أو أوراق مطلوبة\n"
    "- PROCEDURE: يسأل عن خطوات أو كيفية القيام بشيء\n"
    "- COST: يسأل عن رسوم أو مبالغ أو تكاليف\n"
    "- DEADLINE: يسأل عن مدة الإنجاز أو الوقت اللازم\n"
    "- ELIGIBILITY: يسأل عن شروط التقديم أو من يحق له\n"
    "- LEGAL: يسأل عن عقوبات أو جزاءات قانونية فقط\n"
    "- COMPARISON: يسأل عن الفرق بين إجراءين أو وثيقتين\n"
    "- OUT_OF_SCOPE: لا علاقة له بالخدمات الإدارية المغربية\n\n"
    "قواعد صارمة:\n"
    "1. أعطِ كل النوايا المنطبقة — التعدد طبيعي ومتوقع\n"
    "2. LEGAL و OUT_OF_SCOPE لا تُجمع مع نوايا أخرى أبداً\n"
    "3. complexity=MULTIHOP عند وجود نيتين أو أكثر\n"
    "4. hop_count = عدد المعلومات المطلوبة المستقلة (1-4)\n"
    "5. أخرج JSON فقط — بدون أي شرح أو مقدمة"
)

CLASSIFIER_SYSTEM_FR = (
    "Vous êtes un classificateur de questions pour un système de services publics marocains.\n"
    "Une question peut avoir PLUSIEURS intentions simultanément.\n\n"
    "Définitions des intentions:\n"
    "- DOCUMENTS: demande quels documents ou papiers sont requis\n"
    "- PROCEDURE: demande comment faire quelque chose, quelles étapes suivre\n"
    "- COST: demande les frais, prix ou montants à payer\n"
    "- DEADLINE: demande le délai de traitement ou la durée\n"
    "- ELIGIBILITY: demande qui peut postuler ou quelles conditions remplir\n"
    "- LEGAL: demande les sanctions, pénalités ou conséquences pénales uniquement\n"
    "- COMPARISON: demande la différence entre deux procédures ou documents\n"
    "- OUT_OF_SCOPE: sans rapport avec les services administratifs marocains\n\n"
    "Règles strictes:\n"
    "1. Retournez TOUTES les intentions applicables — la multiplicité est normale\n"
    "2. LEGAL et OUT_OF_SCOPE ne se combinent jamais avec d'autres intentions\n"
    "3. complexity=MULTIHOP quand 2+ intentions ou plusieurs sous-questions\n"
    "4. hop_count = nombre d'informations distinctes nécessaires (1-4)\n"
    "5. JSON uniquement — sans explication ni préambule"
)

CLASSIFIER_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "intent_classifier",
        "schema": {
            "type": "object",
            "properties": {
                "intents": {"type": "array", "items": {"type": "string"}},
                "complexity": {"type": "string", "enum": ["SIMPLE", "MULTIHOP"]},
                "hop_count": {"type": "integer"},
            },
            "required": ["intents", "complexity", "hop_count"],
        },
    },
}

# ── Refusal / abstain messages ────────────────────────────────────────────────

REFUSAL = {
    "Darija": "سمحلي، هاد السؤال خارج من اختصاصي. أنا متخصص غير في الخدمات العامة المغربية. يمكنك تزور service-public.ma",
    "arabic_msa": "عذراً، هذا السؤال خارج نطاق اختصاصي. أنا متخصص في الخدمات العامة المغربية فقط. يمكنكم زيارة service-public.ma",
    "french": "Désolé, cette question est hors de mon domaine. Je suis spécialisé dans les services publics marocains. Visitez service-public.ma",
}

ABSTAIN = {
    "Darija": "ماجدتش معلومات كافية فقاعدة البيانات باش نجاوبك. من فضلك تواصل مع المصلحة المختصة أو زور service-public.ma",
    "arabic_msa": "لم أجد معلومات كافية في قاعدة البيانات للإجابة على سؤالكم. يرجى التواصل مع المصلحة المختصة أو زيارة service-public.ma",
    "french": "Informations insuffisantes dans notre base de données. Veuillez contacter le service concerné ou visiter service-public.ma",
}

FALLBACK = {
    "Darija": "كاين مشكل تقني. عاود المحاولة من فضلك.",
    "arabic_msa": "حدث خطأ تقني. يرجى المحاولة مرة أخرى.",
    "french": "Une erreur technique s'est produite. Veuillez réessayer.",
}

LEGAL_DISCLAIMER = {
    "Darija": "\n\n⚠️ تنبيه قانوني: هاد المعلومات للإرشاد العام غير ولا تعوض الاستشارة القانونية.",
    "arabic_msa": "\n\n⚠️ تنبيه قانوني: هذه المعلومات للإرشاد العام فقط ولا تعوض الاستشارة القانونية المتخصصة.",
    "french": "\n\n⚠️ Avertissement légal: Ces informations sont indicatives et ne remplacent pas un conseil juridique professionnel.",
}

NOT_FOUND_LABELS = {
    "DOCUMENTS": {"arabic_msa": "الوثائق", "Darija": "الوثائق", "french": "documents"},
    "COST": {"arabic_msa": "الرسوم", "Darija": "الثمن", "french": "frais"},
    "DEADLINE": {"arabic_msa": "المدة", "Darija": "المدة", "french": "délai"},
    "PROCEDURE": {"arabic_msa": "الخطوات", "Darija": "الخطوات", "french": "étapes"},
    "ELIGIBILITY": {"arabic_msa": "الشروط", "Darija": "الشروط", "french": "conditions"},
    "LEGAL": {"arabic_msa": "العقوبات", "Darija": "العقوبات", "french": "sanctions"},
    "COMPARISON": {"arabic_msa": "المقارنة", "Darija": "المقارنة", "french": "comparaison"},
}


_PLAN_TOOL_DESCRIPTIONS = (
    "retrieve_kb:        standard hybrid BM25+dense search — use for most queries\n"
    "lookup_article:     find a specific law article by number — use when you know المادة N\n"
    "calculate_deadline: find processing time chunks + compute actual date\n"
    "check_eligibility:  find age/status/nationality conditions\n"
    "search_by_amount:   find fee/fine amounts in dirhams"
)

_PLAN_JSON_EXAMPLE = (
    '{"steps": [{"step_id": 1, "intent": "DOCUMENTS", '
    '"sub_question": "...", "tool": "retrieve_kb", '
    '"tool_args": {"query": "..."}, "rationale": "..."}, ...]}'
)


def plan_prompt(question: str, intents: list, language: str) -> str:
    if language in ("arabic_msa", "Darija"):
        return (
            "أنت مخطط لنظام إجابة على أسئلة الخدمات الحكومية المغربية.\n"
            "مهمتك: إنشاء خطة تنفيذ JSON لهذا السؤال.\n\n"
            f"السؤال: {question}\n"
            f"النوايا المكتشفة: {intents}\n\n"
            "الأدوات المتاحة:\n"
            f"{_PLAN_TOOL_DESCRIPTIONS}\n\n"
            f'أنشئ خطة JSON: {_PLAN_JSON_EXAMPLE}\n\n'
            "قواعد: خطوة واحدة لكل نية — JSON فقط\nJSON:"
        )
    return (
        "Vous êtes un planificateur pour un système de QA des services publics marocains.\n"
        "Mission: créer un plan d'exécution JSON pour cette question.\n\n"
        f"Question: {question}\n"
        f"Intentions détectées: {intents}\n\n"
        "Outils disponibles:\n"
        f"{_PLAN_TOOL_DESCRIPTIONS}\n\n"
        f'Créez un plan JSON: {_PLAN_JSON_EXAMPLE}\n\n'
        "Règles: Une étape par intention — JSON uniquement\nJSON:"
    )


def reflect_prompt(sub_question: str, intermediate: str, context_preview: str, language: str) -> str:
    if language in ("arabic_msa", "Darija"):
        return (
            f"السؤال: {sub_question}\n"
            f"الجواب المُنتَج: {intermediate[:200]}\n"
            f"مقتطف من الوثيقة: {context_preview}\n\n"
            "هل الجواب كامل ويغطي السؤال بشكل كافٍ؟\n"
            "أجب بكلمة واحدة فقط: complete أو partial أو not_found"
        )
    return (
        f"Question: {sub_question}\n"
        f"Réponse générée: {intermediate[:200]}\n"
        f"Extrait du document: {context_preview}\n\n"
        "La réponse couvre-t-elle complètement la question?\n"
        "Répondez avec un seul mot: complete ou partial ou not_found"
    )


def not_found_message(intent: str, language: str) -> str:
    lang_key = language if language in ("arabic_msa", "Darija", "french") else "french"
    label = NOT_FOUND_LABELS.get(intent, {}).get(lang_key, intent)
    if language == "Darija":
        return f"المعلومات على {label} ماكاينةش فالوثائق المتاحة."
    if language == "arabic_msa":
        return f"معلومات {label} غير موجودة في الوثائق المتاحة."
    return f"Informations sur les {label} non disponibles dans les documents fournis."
