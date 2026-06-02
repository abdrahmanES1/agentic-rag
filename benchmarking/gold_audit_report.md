# Gold Dataset Audit Report

- Total items: **624**
- Flagged items: **29**

## Issue legend

| Code | Meaning |
|---|---|
| LANG_FIELD_MISMATCH | language != expected_language |
| Q_HAS_ARABIC | Arabizi/French question contains Arabic script |
| Q_WRONG_SCRIPT | Arabic/Darija question is in Latin |
| A_IS_ARABIC | answer is Arabic but should be Latin (Arabizi/French) |
| A_MIXED_SCRIPT | answer mixes scripts |
| A_IS_LATIN | answer is Latin but should be Arabic |
| KW_HAS_ARABIC | keywords contain Arabic but should be Latin |
| KW_IS_LATIN | keywords are Latin but should be Arabic |
| KW_EMPTY | no keywords |
| ARABIZI_IPA_IN_Q/A | IPA diacritics (ā/ḥ) instead of standard Arabizi |
| ANSWER_EMPTY | non-OUTSCOPE with empty answer |
| ANSWER_LOOKS_ABSTAIN | non-OUTSCOPE answer looks like an abstention |
| OUTSCOPE_NOT_ABSTAINING | should_abstain but answer gives content |
| MAYBE_NOT_SELF_CONTAINED | dangling 'this licence/procedure' reference |


## MAYBE_NOT_SELF_CONTAINED  (23 items)

### #1 · french · SIMPLE
- **Q:** Quel est le délai nécessaire pour procéder au règlement du revenu de substitution permanent et quel est le coût associé à cette procédure ?
- **A:** Le règlement du revenu de substitution permanent doit être effectué dans un délai de 30 jours à compter de la réception du certificat de guérison ou 30 jours à compter de la réception du procès-verbal de conciliation signé par l'intéressé, et le coût est gratuit.
- **KW:** ['30 jours', 'gratuit', 'certificat de guérison', 'procès-verbal de conciliation']
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #2 · Darija · DARIJA
- **Q:** واش خاصني نعرف المدة ديال تصفية الإيراد العمري الدائم، وشنو الثمن اللي غادي يخرج عليا باش ندير هاد الإجراء؟
- **A:** خاص يتم التصفية في أجل 30 يوما من تاريخ التوصل بشهادة الشفاء أو 30 يوما من تاريخ التوصل بمحضر الصلح الموقع من طرف المعني بالأمر، والتكلفة هي بالمجان.
- **KW:** ['30 يوما', 'بالمجان', 'شهادة الشفاء', 'محضر الصلح']
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #3 · Arabizi · ARABIZI
- **Q:** chno hia lwaqt li kaykhassni bach itsaffa l'irad l'3omri d-da2im, w chno taklfa dyal had l'ijra2?
- **A:** khass tetm tasfiya f ajal 30 youm mn tarikh tawassol b chahadat chchifa2 wla 30 youm mn tarikh tawassol b mahdar ssol7 lmwaqqa3 mn taraf lma3ni b l2amr, w taklfa hiya b lmajjan.
- **KW:** ['30 youm', 'b lmajjan', 'chahadat chchifa2', 'mahdar ssol7']
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #93 · Darija · DARIJA
- **Q:** شنو هي الوراق اللي خاصني باش ندي رخصة المشاركة فالمعارض، وكم المدة ديال هاد الإجراء؟
- **A:** خاصك تجيب وثيقتين: 1. استمارة التسجيل و 2. نسخة من الوثيقة المثبتة لدفع مبلغ المساهمة. والمدة المحددة هي 20 يوما.
- **KW:** ['استمارة التسجيل', 'نسخة من الوثيقة المثبتة لدفع مبلغ المساهمة', '20 يوما']
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #151 · french · SIMPLE
- **Q:** Quel est le coût et le délai pour obtenir une demande de renouvellement du permis final d'exploitation des installations de production d'électricité à partir de sources d'énergie renouvelable, et quelle est l'autorité responsable de cette procédure ?
- **A:** Le coût est gratuit, et la durée est de 30 jours. L'autorité responsable de cette procédure est le Ministère de la Transition Énergétique et du Développement Durable-Secteur de la Transition Énergétique.
- **KW:** ['gratuit', '30 jours', 'Ministère de la Transition Énergétique et du Développement Durable']
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #152 · Darija · DARIJA
- **Q:** شنو هي التكلفة والمدة ديال تجديد الترخيص النهائي لاستغلال منشآت الطاقة المتجددة، وشنو هي الوزارة المسؤولة على هاد الإجراء؟
- **A:** الثمن هو بلاش، والمدة هي 30 يوم. الجهة المسؤولة على هاد الإجراء هي وزارة الانتقال الطاقي والتنمية المستدامة-قطاع الانتقال الطاقي.
- **KW:** ['بالمجان', '30 يوما', 'وزارة الانتقال الطاقي والتنمية المستدامة']
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #157 · french · SIMPLE
- **Q:** Quels sont les documents requis pour changer le numéro de compte bancaire auprès de la CNSS ? Quel est le coût et le délai de cette procédure ?
- **A:** Pour changer le compte bancaire, il faut fournir : une copie de la carte d'inscription à la CNSS, une copie de la carte nationale d'identité du preneur, et un chèque original tamponné ou un certificat bancaire original pour le compte bancaire. Le coût est gratuit et le délai est de 30 jours.
- **KW:** ["copie de la carte d'inscription à la CNSS", "carte nationale d'identité du preneur", 'chèque original tamponné', 'gratuit', '30 jours']
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #158 · Darija · DARIJA
- **Q:** شنو هي الوراق اللي خاصني باش نبدل رقم الحساب البنكي ديالنا عند الصندوق الوطني لمنظمات الاحتياط الاجتماعي؟ وشنو الثمن والمدة ديال هاد الخدمة؟
- **A:** خاصك تجيب: نسخة من بطاقة التسجيل بالصندوق و.م.ح.ج، نسخة من البطاقة الوطنية للتعريف الخاصة بالمؤمن، وشيك أصلي مشطب عليه أو شهادة بنكية اصلية للحساب البنكي. التكلفة مجانية والمدة هي 30 يوماً.
- **KW:** ['نسخة من بطاقة التسجيل بالصندوق و.م.ح.ج', 'البطاقة الوطنية للتعريف الخاصة بالمؤمن', 'شيك أصلي مشطب عليه', 'مجانا', '30 يوما']
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #173 · french · SIMPLE
- **Q:** Quels sont les frais et le délai requis pour enregistrer un véhicule agricole motorisé importé de l'étranger, et quelle est l'autorité responsable de cette procédure ?
- **A:** Le coût est calculé selon les droits dus au Trésor Public + les redevances dues à l'Agence Nationale de la Sécurité Routière (300 MAD), et le délai est de 30 jours. L'autorité responsable de cette procédure est l'Agence Nationale de la Sécurité Routière.
- **KW:** ['Trésor Public', 'Agence Nationale de la Sécurité Routière', '300 dirhams', '30 jours']
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #202 · french · SIMPLE
- **Q:** Quels sont le coût et le délai requis pour supprimer une activité commerciale en cas de pluralité d'activités exercées au même local, et quels sont les documents essentiels nécessaires pour mener à bien cette procédure ?
- **A:** Le coût de l'inscription au registre du commerce est de 50 dirhams, et la durée estimée est de deux jours. Les documents requis comprennent : un certificat original d'inscription au registre professionnel, et le formulaire de déclaration modèle 4 rempli et signé par l'obligé ou son mandataire muni d
- **KW:** ['50 dirhams', 'deux jours', "certificat original d'inscription au registre professionnel", 'formulaire de déclaration modèle 4']
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #203 · Darija · DARIJA
- **Q:** شنو هي التكلفة والمدة ديال حذف نشاط تجاري فاش كاين بزاف الأنشطة بنفس البلاصة، وشنو هما الوثائق الأساسية اللي خاصني باش نكمل هاد الإجراء؟
- **A:** الرسم ديال التسجيل بالسجل التجاري هو 50 درهما، والمدة هي يومان. الوثائق المطلوبة تشمل: شهادة أصلية للتسجيل بالرسم المهني، والتصريح نموذج 4 محرر وموقع من طرف الملزم أو وكيله المزود بوكالة كتابية.
- **KW:** ['50 درهما', 'يومان', 'شهادة أصلية للتسجيل بالرسم المهني', 'التصريح نموذج 4']
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #209 · Darija · DARIJA
- **Q:** شنو هي الوراق اللي خاصني باش ندير دفع دخول الاستثمارات، وكم كيكلف و شحال المدة ديال هاد الإجراء؟
- **A:** باش دير دفع دخول الاستثمارات، خاصك تجيب وثائق بحال: ملحق بنكي معبأ وموقع ومختوم، ونسخة من فواتير الفوائد. التكلفة هي بالمجان والمدة ديال هاد الإجراء هي 60 يوما.
- **KW:** ['ملحق بنكي معبأ وموقع ومختوم', 'فواتير الفوائد', '60 يوما']
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #295 · arabic_msa · SIMPLE
- **Q:** ما هي التكلفة والمدة الزمنية للحصول على الاعتماد أو تجديده لمراقبة أجهزة الرفع ماعدا المصاعد ورافعات الأثقال، وما هي الجهة المستقبلة لهذا الطلب؟
- **A:** التكلفة هي بالمجان، والمدة هي 60 يوماً. يجب توجيه الطلب إلى وزارة الشغل والإدماج المهني.
- **KW:** ['بالمجان', '60 يوما', 'وزارة الشغل والإدماج المهني']
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #301 · french · SIMPLE
- **Q:** Quel est le coût et la durée pour obtenir l'allocation de décès destinée aux retraités assurés, et quelle est l'autorité responsable de cette procédure ?
- **A:** Le coût est gratuit, la durée est de 60 jours, et l'autorité responsable est la Direction de la Défense Nationale.
- **KW:** ['gratuit', '60 jours', 'Direction de la Défense Nationale']
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #305 · french · SIMPLE
- **Q:** Quels sont le coût et le délai pour obtenir une copie des extraits du Registre de Commerce Central, et quels documents sont nécessaires pour cette démarche ?
- **A:** Pour obtenir la copie des extraits délivrée par l'Office Central du Registre de Commerce, les frais s'élèvent à 300 dirhams et le délai est de 3 jours. Les documents requis sont : 1. Le formulaire de demande de copies d'extraits du registre de commerce C4 et 2. La preuve du paiement des droits dus.
- **KW:** ['300 dirhams', '3 jours', "formulaire de demande de copies d'extraits du registre de commerce C4", 'preuve du paiement des droits dus']
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #370 · french · SIMPLE
- **Q:** Quels sont les documents essentiels requis pour conclure un contrat d'alimentation en électricité moyenne tension ? Et quelle est l'entité responsable de cette procédure ?
- **A:** Les documents requis incluent : 1. Le dossier de demande d'abonnement, 2. Une copie de la carte nationale d'identité, et 3. Une copie de la carte de résidence ou du passeport. L'entité responsable est l'Office National de l'Électricité et de l'Eau Potable - Secteur Électricité.
- **KW:** ["dossier de demande d'abonnement", "copie de la carte nationale d'identité", "Office National de l'Électricité et de l'Eau Potable"]
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #371 · Darija · DARIJA
- **Q:** شنو هي الوراق الأساسية اللي خاصني باش نبرم عقد ديال الكهرباء بجهد متوسط؟ و شكون هو المسؤول على هاد الإجراءات كاملة؟
- **A:** الوراق اللي خاصك كتشمل: 1. طلب الاشتراك، 2. نسخة من بطاقة التعريف الوطنية، و 3. نسخة من بطاقة الإقامة أو جواز السفر. أما الجهة المسؤولة فهي المكتب الوطني للكهرباء والماء الصالح للشرب - قطاع الكهرباء.
- **KW:** ['طلب الاشتراك', 'نسخة من بطاقة التعريف الوطنية', 'المكتب الوطني للكهرباء والماء الصالح للشرب']
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #397 · french · SIMPLE
- **Q:** Quel est le coût et la durée pour obtenir une licence professionnelle limitée de responsable de navigation maritime sur les navires dont la charge totale équivaut ou dépasse 500, et quelle est l'autorité responsable de cette procédure ?
- **A:** Le coût est de 500 dirhams, et la durée est de 7 jours. L'autorité responsable de cette procédure est le Ministère du Transport et de la Logistique.
- **KW:** ['500 dirhams', '7 jours', 'Ministère du Transport et de la Logistique']
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #520 · Darija · DARIJA
- **Q:** شنو هي الوراق اللي خاصني باش ندير أداء برسم صفقة بالدرهم عن طريق الخصم من حساب "خاص"، وشنو الثمن والمدة ديال هاد الخدمة؟
- **A:** الوراق اللي خاصك كتشمل: ملحق بنكي معبأ وموقع ومختوم، نسخة من عقد الصفقة، ونسخة من كشوفات الحساب الخاص. الثمن هو بلاش، والمدة هي 60 يوم.
- **KW:** ['ملحق بنكي معبأ وموقع ومختوم', 'عقد الصفقة', 'كشف الحساب الخاص', 'بالمجان', '60 يوما']
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #529 · french · SIMPLE
- **Q:** Quel est le coût et le délai pour effectuer le paiement des frais d'élimination de marchandises à l'étranger, et quels sont les cinq documents requis pour cette procédure ?
- **A:** Le coût est gratuit, et la durée de la procédure est de 60 jours. Les documents requis sont : 1. Preuve du transfert des revenus d'exportation vers le Maroc, si nécessaire 2. Copie de l'extrait du registre de commerce (formulaire 7 et/ou certificat d'identification unifié de l'entreprise) 3. Copie d
- **KW:** ['gratuit', '60 jours', "Preuve du transfert des revenus d'exportation vers le Maroc", "extrait du registre de commerce (formulaire 7 et/ou certificat d'identification unifié de l'entreprise)", "Déclaration unifiée des marchandises à titre d'exportation"]
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #572 · Darija · DARIJA
- **Q:** شنو هي الوراق اللي خاصني باش ندير أداء فوائد التأخير على فواتير المساعدة التقنية، وشنو الثمن والمدة ديال هاد الإجراء؟
- **A:** باش دير أداء فوائد التأخير على فواتير المساعدة التقنية، خاصك تجيب: 1. ملحق بنكي معبأ وموقع ومختوم، 2. نسخة من الفواتير، 3. كيفية حساب فوائد التأخير، 4. عقد ينص على أداء فوائد التأخير، و5. نسخة من مستخرج السجل التجاري (نموذج 7 و/ أو شهادة التعريف الموحد للمقاولة). التكلفة هي بالمجان والمدة هي 60 يوما
- **KW:** ['بالمجان', '60 يوما', 'ملحق بنكي معبأ وموقع ومختوم', 'نسخة من مستخرج السجل التجاري']
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #591 · french · SIMPLE
- **Q:** Quel est le coût et le délai pour effectuer l'opération de paiement du rendement de liquidation, et quel est un document essentiel requis pour cette procédure ?
- **A:** Le service d'exécution du rendement de liquidation est gratuit, dure 60 jours. Parmi les documents essentiels requis figure le certificat de décharge fiscale.
- **KW:** ['Office des Changes', 'gratuit', '60 jours', 'certificat de décharge fiscale']
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']

### #594 · arabic_msa · MULTIHOP
- **Q:** إذا كانت الشركة ستقوم بالتصفية، فما هي الوثائق التي يجب تقديمها لإجراء أداء محصول التصفية، وما هو الإجراء المتعلق بتسجيل الشركة الذي يمكن أن يكون مرتبطاً بهذه العملية؟
- **A:** لإجراء أداء محصول التصفية، يجب تقديم وثائق مثل محضر الجمع العام للمساهمين أو نسخة من المقرر القضائي الذي يقرر حل الشركة أو إخضاعها للتصفية الجزئية أو الكلية، بالإضافة إلى شهادة الإبراء الضريبي. ومن الإجراءات ذات الصلة تسجيل شركة مساهمة بالسجل التجاري.
- **KW:** ['محضر الجمع العام للمساهمين', 'شهادة الإبراء الضريبي', 'تسجيل شركة مساهمة بالسجل التجاري']
- **all issues:** ['MAYBE_NOT_SELF_CONTAINED']


## ANSWER_LOOKS_ABSTAIN  (4 items)

### #43 · Arabizi · ARABIZI
- **Q:** Wach khassni n3raf chno hiya lwaraq li khasni bach njib qarar lmwafaqa ttiqniya l'istirad fhoul l2abqar mn ssoulalat l2asila, w chno hia taklfa w lmoda dyal had chi?
- **A:** bach takhod l9arar, khassek tqaddem: 1. talab l2istirad 2. rokhsat istabl l7ajr ss7i li tahaddad taa9a listi3abiya 3. liltizam b i7tiram chchorot lm7addada f daftar tta7ammolat 4. la2i7at lmostafidin mn 3amaliyat l2istirad ssab9a 5. ssijill ttijari bnnisba l charikat wla l9anoun l2asasi bnnisba l ta
- **KW:** ['talab l2istirad', 'rokhsat istabl l7ajr ss7i li tahaddad taa9a listi3abiya', 'b lmajjan', '15 youm']
- **all issues:** ['ANSWER_LOOKS_ABSTAIN']

### #47 · Arabizi · ARABIZI
- **Q:** Chno hiya lwaraq li khassni bach nakhod l9arar dyal tawafuq ttiqni bach nstired fhoul lma3z mn ssoulalat l2asila, w ch7al kaytsehhel?
- **A:** Khassek tqaddem: 1. talab l2istirad 2. rokhsat istabl l7ajr ss7i li tahaddad taa9a listi3abiya 3. liltizam b i7tiram chchorot lm7addada f daftar tta7ammolat 4. la2i7at lmostafidin mn 3amaliyat l2istirad ssab9a 5. ssijill ttijari bnnisba l charikat wla l9anoun l2asasi bnnisba l ta3awoniyat w jam3iyat
- **KW:** ['talab l2istirad', 'rokhsat istabl l7ajr ss7i li tahaddad taa9a listi3abiya', 'b lmajjan', '15 youm']
- **all issues:** ['ANSWER_LOOKS_ABSTAIN']

### #141 · arabic_msa · MULTIHOP
- **Q:** إذا كان المرء يحتاج إلى إتمام إجراء الحصول على معطيات جيوديزية، فما هي الوثائق التي يجب أن يمتلكها أولاً لإكمال إجراء البطاقة الوطنية للتعريف الإلكترونية؟
- **A:** لإجراء البطاقة الوطنية للتعريف الإلكترونية، لا تذكر النصوص وثائق مطلوبة مسبقًا قبل البدء في الحصول على معطيات جيوديزية، ولكن للحصول على المعطيات الجيوديزية نفسها، يجب تقديم طلب المعطيات الجيوديزية وبطاقة التعريف الوطنية.
- **KW:** ['الحصول على معطيات جيوديزية', 'البطاقة الوطنية للتعريف الإلكترونية', 'طلب المعطيات الجيوديزية', 'بطاقة التعريف الوطنية']
- **all issues:** ['ANSWER_LOOKS_ABSTAIN']

### #292 · Arabizi · ARABIZI
- **Q:** Wach kayn délai w chi taklifa khassni bach ndir l'Amr bel Ada' wla nrfa3 lyad b t3wid 3la naz3 lmilkiya l lmanfa3a l3amma, w chno huma lwata2i9 l2asasiyin li khasni?
- **A:** lmoda dyal l3amaliya hiya 60 youm w taklfa b lmajjan. mn lwata2i9 lmatluba: talab maktoub, chahadat lmilkiya (7sab naw3 tta3wid), w rasm lmilkiya l l3aqarat lli machi m7affda.
- **KW:** ['b lmajjan', '60 youm', 'talab maktoub', 'chahadat lmilkiya']
- **all issues:** ['ANSWER_LOOKS_ABSTAIN']


## OUTSCOPE_NOT_ABSTAINING  (2 items)

### #189 · arabic_msa · OUTSCOPE
- **Q:** ما هي الإجراءات المتبعة لتغيير التوطين البنكي إذا كان المؤمن له لا يزال مقيماً داخل المغرب ويريد فقط تغيير بنكه؟
- **A:** عذراً، هذه الوثيقة تتناول إجراء تغيير التوطين البنكي الخاص ب "مؤمن له" بالخارج ولا تغطي الإجراءات المتبعة للمقيمين داخل المغرب.
- **KW:** ['تغيير التوطين البنكي', 'بالخارج']
- **all issues:** ['OUTSCOPE_NOT_ABSTAINING']

### #257 · arabic_msa · OUTSCOPE
- **Q:** ما هي الجهة المسؤولة عن إصدار شهادة عدم الإيجار إذا كان المعني بالطلب هو موظف في وزارة التربية الوطنية بدلاً من إدارة الدفاع الوطني؟
- **A:** لا يمكن تحديد الجهة المسؤولة لإصدار شهادة عدم الإيجار لموظف في وزارة التربية الوطنية بناءً على النص المقدم، حيث أن الجهة المسؤولة المذكورة هي إدارة الدفاع الوطني.
- **KW:** ['إدارة الدفاع الوطني']
- **all issues:** ['OUTSCOPE_NOT_ABSTAINING']

