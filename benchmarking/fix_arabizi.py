# -*- coding: utf-8 -*-
"""
Fix 50 Arabizi questions in benchmark_testset_gold.json:
  - Remove embedded Arabic script from Arabizi questions
  - Replace IPA/academic diacritics with standard Moroccan Arabizi
"""
import json
from pathlib import Path

GOLD = Path(__file__).parent / "benchmark_testset_gold.json"

# Corrections keyed by position in the Arabizi-only list (0-indexed)
FIXES = {
    0:   "chno hia lwaqt li kaykhassni bach itsafa l'irad l'omri d-daim, w chno tklfa dyal had l'ijra'?",
    3:   "Wach khassni nkhdem chi lwaraq bach njib tarkhis lziyara lmwaq3 athariya l'groupe d'etudiants, w chno huma jihat mas'oula 3la had shi?",
    7:   "Wach khassni n3raf chno hiya lwaraq li khasni bach njib qarar lmwafaqa ttiqniya l'istirad d'fhoul dial lbqar men slalat asila, w chno hia taklifa w mdda dyal had shi?",
    10:  "Wach khassni nji b'lwaraq achmen hwayej bach njib nazir chahada tibbiya dyal wafa mn lmostachfayat ljami3iya, wach kayn chi taklifa wla ch7al lmudda?",
    13:  "Wach khassni n3raf chno hia lwaraq li khasni bach njib izin lwolouj lmelk lghabawi, w ch7al kaykallef w ch7al lwaqt?",
    14:  "Wach khassni njawjed chno hia lwaraq li khasni, wchno howa tklfa w lwaqt dyal ittifaqiyat shraka bin lmouassasat ttaqliwiya w chouraka?",
    15:  "Wach khassni n3raf chno hia lwaraq li khasni bach njib rkhssat lhtilal lmou'aqqaf lmelk l'aam bla bina, w chno hiya tklfa w lmudda?",
    17:  "Wach khassni nji b'chi waraq bach nsajjel qabli f'aqsam tahdiyar chahadat attaqni l'3ali, w chno hia lkost w lwaqt?",
    19:  "Wach khassni nji b'chno bach njib chahada dyal lma3ach men sandouq lmaghribi lttaqa3ud, w chno hia tklfa w lwaqt?",
    20:  "Wach khassni nji b'lwaraq achmen 7wayej bach njib jwaz safar biometrique, w chno lkost w lwaqt?",
    28:  "Wach khassni nbadl rraqm dyal lhsab lbnki 3and CNSS? Chno hiya lwaraq li khasni, w chno sman w chno mdda?",
    31:  "Shno hiya taklifa w lwaqt li khassni bach nsajjel merkeba fala7iya b'mouharrek moustawrada men lbar, w ach hiya jiha mas'oula 3la had chi?",
    32:  "Wach kayn chi deadline wla cout wela lwaraq li khassni bach ntlab tasfiya l'irad l'omri d-daim dial dahaya d'hawadith chaghol li fihom 3ajz joz'i dyal da3m kaytzid 3la 10%?",
    33:  "Chno hia tklfa w lmudda dyal rapport technique bach nmtabeqou wasila dyal nqal lmasadir lmach3a, w chno homa les documents li khassni njma3?",
    35:  "Chno hiya taklfa w lwaqt li kaytkhassos l7df dyal nachat tijari fach kayn bzaf l'anachita f'nafs lmahal, w chno homa lwathayeq l'asasiya li khassni bach nkammel had l'ijra'?",
    38:  "Wach khassni nji b'lwaraq achmen haja bach njib chahada dyal l'istighllal lfalahi men lmaktab ljihawi lil'ithimar lfalahi Tadla, w chno lkost w ch7al lwaqt?",
    41:  "Wach khassni nji b'ach lwaraq, w chno lkost w ch7al ghadi yakhdou liya bach njib chahada tarqim dyal khlaya nahl?",
    42:  "Wach khassni nji b'lwaraq chno hiya bach njib chahada tawjih lnissa w l'atfal dahaya dl'ounf, w ch7al tkostaha w ch7al lwaqt?",
    43:  "Wach khassni nji b'lwaraq achmen 7wayej bach njib chahada 3adam l'ijar, w chno lkost w ch7al lwaqt?",
    45:  "chno hia lwaraq li khassni bash nqaddem dossier dyal qarar lmwafaqa lbi2iya, w chno hiya la période dyal lkhadma?",
    46:  "Wach khassni n3raf chno hia lwaraq w kamel chi haja bach njib lmwafaqa ssi7iya dyal tasdir lkhyoul? Wach kayn chi taklifa w ch7al lwaqt?",
    47:  "Wach khassni nji b'lwaraq chno hiya bach njib mn mahdar qaboul chakhes mtawafi f'lmostachfiyat ljihawiya w l'iqlimiya, w chno lkost w lwaqt?",
    48:  "Wach kayn délai w chi taklifa khassni bach ndir l'Amr bel Ada' aw nrfa3 l'yad b't3wid 3la naz3 lmelkiya dyal lmanfa3a l3amma, w chno homa lwathayeq l'asasiya li khasni?",
    49:  "Chno hiya taklfa w lwaqt li kaykhass ykoun bach nakhd l'iktimad aw njaddedouh lmouqabaqa dyal jihat rraf3 ma3adi ssama3a w rfa3at l'ath9al, w ach hiya jiha li khassni nssifet liha ttalab?",
    51:  "Chno thaman w chhal lmoda bach nakhod noskhit ttaqyidat dyal ssijil ttijari lmarkazi, w achmen wathayeq khassni njawjed?",
    53:  "Wach khassni njiw b'lwaraq, w chno hia lkost w ch7al lwaqt bach ndir rapport d'expertise technique f prévention des radiations?",
    54:  "Wach khassni nssajel 3alama tijariya f'l'Office Marocain de la Propriété Industrielle et Commerciale, wach kayn waqt w chno lwaraq?",
    58:  "Wach khassni njiw b'lwaraq chno hiya bach ndir l'intiqal ljami3i, w ch7al ghayakhdem liya?",
    61:  "Wach khassni nji b'ay documents bach nkhdm 3qd ttazwid bttaqa lkahrabaiya dhat ljahd l3ali, w chno lwaqt li kayakhdem fih had l'procedure?",
    64:  "Shno hiya tklfa w lmoda dyal l'ijaza ahliya dyal rbabin ssofon fihom hamoula ijmaliya taqell men 500 w katsamrou nhda ssahel, w chno ahmm wathayeq khassni nwajed?",
    66:  "chno hia taklfa w chhal lmoda bach nkhd l'ijaza ahliya d-dabit mas2oul 3la lkhafara lmilhiya lssofra li hamoulatoha aqal men 500, w chno huma ahamm les documents li khassni nwjed?",
    68:  "Wach khassni nji b'ay lwaraq bach njib l'ijaza l'ahliya d-dabit awwal ghir mahduda, w chno hiya tklfa w lwaqt?",
    69:  "Shno hiya taklfa w lwaqt bach nkhd l'ijaza l'ahliya d-dabit mas2oul 3la lkhaffara lhandasiya, w chno huma ahamm lwathayeq lmatlouba?",
    70:  "Chno hiya taklfa w lwaqt bach nkhd l'ijaza l'ahlia d-dabit mouhandis thani ghir mahduda, w chno huma ahamm lwathayeq lmatlouba?",
    73:  "Wach khassni nji b'ay lwaraq bach njib ijazat moraqib jawwi, w chno hia tklfa w kaml lwaqt?",
    75:  "Wach khassni nji b'lwaraq achmen 7wija, w chno lmdda w tklfa bach ndir training f'L'Office National Marocain du Tourisme?",
    78:  "Wach khassni n3raf chno hia lwaraq li khasni bach ndir adaa ajour l'artistes men taraf jiha khassa, w chno mdada?",
    79:  "Wach khassni n3raf chno hia lwaraq li khasni bach ndir ada' tta3widat brasm tqdim lkhadamat, w chno lwaqt w tklfa?",
    80:  "Wach khassni n3raf chno lwaraq bach ndir ada' lhouqouq d-dawriya dyal kira ttayara lchi jiha machi charikat tayaran, w chno mdada w chi tklfa?",
    83:  "Wach kayn chi taklifa wla mdada lwaqtiya bach nqaddem talab istirja3 masarif ssafar dyal ajnabi li khadamin f'lmaghrib, wchno hiya lwathayeq l'asasiya?",
    85:  'Wach khassni nji b\'lwaraq chno hiya bach ndir ada\' brasm safqa mharara bkamiliha bdirham 3an tariq lkhasm mn hsab "khas"? Wach kayn tklfa w wach kayn mdda?',
    86:  "Wach khassni nqaddem chno men lwaraq bach ndawi 3la ta3widat 3n adrare rsif mina f'lkharej, w chno lwaqt w tklfa?",
    88:  "Wach khassni nji b'lwaraq chno hiya bach ndir ada' hissis inbi3at lgaz l'anbi3 lharari lcharikat ttayaran, w chno mdadda w tklfa?",
    89:  "Wach khassni nkhdem chno lwaraq wla chno tklfa w ch7al lwaqt bach ndfa3 dfaa brasm 3amaliyat istirad?",
    91:  "Wach khassni nji b'lwaraq chno hiya bach ndir l'importation 3la 7sab tghayyer smit fournisseur ajnabi, w chno hia lwaqt w lkost?",
    94:  "chno hia lwaraq li khassni bash nkhallas qard b'iqtita3 mn mahsoul ttasdir, w chno mdada w tklfa?",
    95:  "Wach khassni n3raf chno hia lwaraq li khasni bach ndir ada' qimat istirad kan msarrah bih mabda2iyan bghayri ada', w chno hiya lwaqt w tklifa?",
    97:  "Wach khassni n3raf chno hia lwaraq li khasni bach ndir ada' mahsoul bay3 3qar lda maktab ssarf, w chno hiya tklfa w lwaqt?",
    99:  "Wach khassni nji b'lwaraq chno w lwaqt bach ndir sarf dyal masarif mokhtalifa machi mdkoura f'twjihat l3amma?",
    101: "Wach khassni nkhdem chno lwaraq bach ndawer masarif l'iqama f compte banki mkhassas men taraf ljami3a, wchno tklfa w lmoudda?",
}

def has_arabic(text):
    return any('؀' <= c <= 'ۿ' for c in text)

def has_ipa(text):
    ipa = set('āēīōūḥḍṭẓṣġʿʾ')
    return any(c in ipa for c in text.lower())

with open(GOLD, encoding='utf-8') as f:
    gold = json.load(f)

arabizi_indices = [i for i, item in enumerate(gold) if item.get('language') == 'Arabizi']

fixed = 0
for arabizi_pos, new_q in FIXES.items():
    gold_idx = arabizi_indices[arabizi_pos]
    old_q = gold[gold_idx]['question']
    gold[gold_idx]['question'] = new_q
    fixed += 1

# Verify no more bad items
remaining_bad = 0
for i in arabizi_indices:
    q = gold[i]['question']
    if has_arabic(q) or has_ipa(q):
        remaining_bad += 1
        print(f"  Still bad: [{i}] {q[:80]}")

with open(GOLD, 'w', encoding='utf-8') as f:
    json.dump(gold, f, ensure_ascii=False, indent=2)

print(f"Fixed {fixed} questions. Remaining bad: {remaining_bad}. Saved.")
