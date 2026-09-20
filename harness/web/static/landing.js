/* Landing and sign-in for the Evaluation Harness. Vanilla JS on purpose: the app is served by
   a stdlib server and ships no framework. Nothing here computes a metric;
   the numbers on this page are the product's own counts. */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const LANGS = ["en", "hi", "mr"];
const LANG_LABELS = { en: "EN", hi: "हि", mr: "म" };

const STR = {
  brand: { en: "Evaluation Harness", hi: "इवैल्युएशन हार्नेस", mr: "इव्हॅल्युएशन हार्नेस" },
  brandSub: { en: "Institutional LLM evaluation", hi: "संस्थागत एलएलएम मूल्यांकन", mr: "संस्थात्मक एलएलएम मूल्यांकन" },
  navReal: { en: "What's real", hi: "क्या वास्तविक है", mr: "काय खरे आहे" },
  navScreens: { en: "Screens", hi: "स्क्रीन", mr: "स्क्रीन" },
  navFoundations: { en: "Foundations", hi: "आधार", mr: "पाया" },
  navConsoles: { en: "Consoles", hi: "कंसोल", mr: "कन्सोल" },
  signIn: { en: "Sign in", hi: "साइन इन", mr: "साइन इन" },
  eyebrow: { en: "INSTITUTIONAL LLM EVALUATION", hi: "संस्थागत एलएलएम मूल्यांकन", mr: "संस्थात्मक एलएलएम मूल्यांकन" },
  h2Prefix: { en: "Prove which model clears your bar,", hi: "साबित करें कि कौन सा मॉडल आपकी सीमा पार करता है,", mr: "कोणते मॉडेल तुमची मर्यादा पार करते ते सिद्ध करा," },
  h2Accent: { en: "with the margin of error on the record.", hi: "त्रुटि सीमा रिकॉर्ड पर सहित।", mr: "त्रुटी मर्यादा नोंदीसह." },
  subCopy: { en: "A model × profile × pass matrix over your own data: every item scored, every rupee of spend metered from the provider's own counts, every failure mode measured on its own, and a paired, corrected test that decides whether the order means anything. Sovereign infrastructure that runs on your machine; the traces stay in your control.",
    hi: "आपके अपने डेटा पर मॉडल × प्रोफ़ाइल × पास मैट्रिक्स: हर आइटम स्कोर, हर खर्च प्रदाता की अपनी गणना से मापा हुआ, हर विफलता अलग से मापी हुई, और एक युग्मित, संशोधित परीक्षण जो तय करता है कि क्रम का अर्थ है या नहीं। आपकी मशीन पर चलने वाला संप्रभु ढाँचा; ट्रेस आपके नियंत्रण में रहते हैं।",
    mr: "तुमच्या स्वतःच्या डेटावर मॉडेल × प्रोफाइल × पास मॅट्रिक्स: प्रत्येक आयटम स्कोअर, प्रत्येक खर्च पुरवठादाराच्या स्वतःच्या संख्येवरून मोजलेला, प्रत्येक अपयश स्वतंत्रपणे मोजलेले, आणि क्रमाला अर्थ आहे का हे ठरवणारी जोडी, दुरुस्त चाचणी. तुमच्या मशीनवर चालणारी सार्वभौम पायाभूत सुविधा; ट्रेस तुमच्या नियंत्रणात राहतात." },
  enterCta: { en: "Enter the Evaluation Harness", hi: "इवैल्युएशन हार्नेस में प्रवेश करें", mr: "इव्हॅल्युएशन हार्नेसमध्ये प्रवेश करा" },
  goToWorkspace: { en: "Go to workspace", hi: "कार्यक्षेत्र पर जाएँ", mr: "कार्यक्षेत्रावर जा" },
  seeReal: { en: "See what's actually real", hi: "देखें क्या वास्तव में वास्तविक है", mr: "प्रत्यक्षात काय खरे आहे ते पहा" },
  statFailure: { en: "failure modes it is built against", hi: "जिन विफलताओं के विरुद्ध बना", mr: "ज्या अपयशांविरुद्ध बांधले" },
  statInvariants: { en: "invariants enforced", hi: "लागू अपरिवर्तनीय नियम", mr: "लागू अपरिवर्तनीय नियम" },
  statCases: { en: "worked case studies", hi: "पूर्ण केस स्टडी", mr: "पूर्ण केस स्टडी" },
  statLanguages: { en: "languages recognised", hi: "मान्य भाषाएँ", mr: "मान्यताप्राप्त भाषा" },
  quote1: { en: "\"Non-significant means non-significant. When the interval straddles zero the harness refuses to name a winner and says", hi: "\"गैर-महत्वपूर्ण का अर्थ गैर-महत्वपूर्ण है। जब अंतराल शून्य के दोनों ओर हो, हार्नेस विजेता घोषित करने से इनकार करता है और बताता है", mr: "\"गैर-लक्षणीय म्हणजे गैर-लक्षणीय. जेव्हा अंतराल शून्याच्या दोन्ही बाजूंना असतो, तेव्हा हार्नेस विजेता ठरवण्यास नकार देते आणि सांगते" },
  quote2: { en: "how many more items it would take.\"", hi: "कितने और आइटम चाहिए होंगे।\"", mr: "आणखी किती आयटम लागतील.\"" },
  quoteTag: { en: "THE HONESTY RULE (INVARIANT I6)", hi: "ईमानदारी का नियम (अपरिवर्तनीय I6)", mr: "प्रामाणिकतेचा नियम (अपरिवर्तनीय I6)" },
  realEyebrow: { en: "WHAT'S ACTUALLY REAL TODAY", hi: "आज वास्तव में क्या वास्तविक है", mr: "आज प्रत्यक्षात काय खरे आहे" },
  realTitle: { en: "Not a mockup: three things genuinely run", hi: "मॉकअप नहीं: तीन चीज़ें वास्तव में चलती हैं", mr: "मॉकअप नाही: तीन गोष्टी खरोखर चालतात" },
  real1t: { en: "A paired, corrected test", hi: "युग्मित, संशोधित परीक्षण", mr: "जोडी, दुरुस्त चाचणी" },
  real1d: { en: "Exact McNemar for binary metrics, paired bootstrap for continuous ones, Holm-Bonferroni across the family. Every ranking shows its interval, and a gap inside it is reported as not separable, with the items needed to settle it.", hi: "बाइनरी मीट्रिक के लिए सटीक मैकनेमार, सतत के लिए युग्मित बूटस्ट्रैप, परिवार भर में होल्म-बोनफेरोनी। हर रैंकिंग अपना अंतराल दिखाती है, और उसके भीतर का अंतर अलग न होने योग्य बताया जाता है, साथ में उसे तय करने के लिए आवश्यक आइटम।", mr: "बायनरी मेट्रिकसाठी अचूक मॅकनेमार, सतत मेट्रिकसाठी जोडी बूटस्ट्रॅप, कुटुंबभर होल्म-बॉनफेरोनी. प्रत्येक क्रमवारी आपला अंतराल दाखवते, आणि त्यातील फरक वेगळा न करता येणारा म्हणून नोंदवला जातो, तो ठरवण्यासाठी लागणाऱ्या आयटमसह." },
  real2t: { en: "Spend metered, not estimated", hi: "खर्च मापा गया, अनुमानित नहीं", mr: "खर्च मोजलेला, अंदाजित नाही" },
  real2d: { en: "Generation, judge, embedding and rerank are billed from the provider's own usage counts into four buckets that sum to the total. An unpriced model fails validation instead of ranking as free.", hi: "जनरेशन, जज, एम्बेडिंग और रीरैंक प्रदाता की अपनी उपयोग गणना से चार बकेट में बिल होते हैं जो कुल में जुड़ते हैं। बिना मूल्य वाला मॉडल मुफ़्त रैंक होने के बजाय सत्यापन में विफल होता है।", mr: "जनरेशन, जज, एम्बेडिंग आणि रीरँक पुरवठादाराच्या स्वतःच्या वापर संख्येवरून चार बकेटमध्ये बिल होतात जी एकूण रकमेत जुळतात. किंमत नसलेले मॉडेल मोफत म्हणून क्रमवारीत येण्याऐवजी पडताळणीत अपयशी ठरते." },
  real3t: { en: "Failure kept apart from wrongness", hi: "विफलता को गलती से अलग रखा", mr: "अपयश चुकीपासून वेगळे ठेवले" },
  real3d: { en: "Truncation, refusal, extraction failure and injection are reported as their own rates, never folded into accuracy. Hidden reasoning tokens are recorded per row so an empty answer reads as a budget finding.", hi: "ट्रंकेशन, इनकार, निष्कर्षण विफलता और इंजेक्शन अपनी दरों के रूप में रिपोर्ट होते हैं, सटीकता में कभी नहीं मिलाए जाते। छिपे तर्क टोकन प्रति पंक्ति दर्ज होते हैं ताकि खाली उत्तर बजट की खोज के रूप में पढ़ा जाए।", mr: "ट्रंकेशन, नकार, निष्कर्षण अपयश आणि इंजेक्शन स्वतःच्या दरांनुसार नोंदवले जातात, अचूकतेत कधीही मिसळले जात नाहीत. लपलेले तर्क टोकन प्रत्येक ओळीसाठी नोंदवले जातात जेणेकरून रिकामे उत्तर बजेटचा निष्कर्ष म्हणून वाचले जाईल." },
  screensEyebrow: { en: "THE SCREENS", hi: "स्क्रीन", mr: "स्क्रीन" },
  screensTitle: { en: "Every screen, and what it is for", hi: "हर स्क्रीन, और वह किसलिए है", mr: "प्रत्येक स्क्रीन, आणि ती कशासाठी आहे" },
  screensDesc: { en: "The same task bar you get once you are signed in. Nothing here is marketed as done when it is not: every screen listed exists and is covered by the offline test suite.", hi: "वही टास्क बार जो साइन इन के बाद मिलता है। यहाँ कुछ भी अधूरा होते हुए पूरा नहीं बताया गया: सूचीबद्ध हर स्क्रीन मौजूद है और ऑफ़लाइन परीक्षण सूट में शामिल है।", mr: "साइन इन केल्यावर मिळणारा तोच टास्क बार. येथे काहीही अपूर्ण असताना पूर्ण म्हणून सांगितलेले नाही: सूचीबद्ध प्रत्येक स्क्रीन अस्तित्वात आहे आणि ऑफलाइन चाचणी संचात समाविष्ट आहे." },
  foundEyebrow: { en: "THE FOUNDATIONS", hi: "आधार", mr: "पाया" },
  foundTitle: { en: "Why an institution can trust it, with the specifics", hi: "एक संस्था इस पर क्यों भरोसा कर सकती है, विवरण सहित", mr: "एखादी संस्था यावर का विश्वास ठेवू शकते, तपशिलासह" },
  found1l: { en: "PBKDF2-HMAC-SHA256, 200,000 iterations", hi: "PBKDF2-HMAC-SHA256, 2,00,000 पुनरावृत्तियाँ", mr: "PBKDF2-HMAC-SHA256, 2,00,000 पुनरावृत्ती" },
  found1d: { en: "the pilot password is hashed before it is ever compared; the plain value is never held for comparison", hi: "पायलट पासवर्ड तुलना से पहले हैश होता है; सादा मान कभी तुलना के लिए नहीं रखा जाता", mr: "पायलट पासवर्ड तुलनेपूर्वी हॅश केला जातो; साधे मूल्य कधीही तुलनेसाठी ठेवले जात नाही" },
  found2l: { en: "HttpOnly session cookies, 12-hour expiry", hi: "HttpOnly सत्र कुकी, 12 घंटे की समाप्ति", mr: "HttpOnly सत्र कुकी, 12 तासांची मुदत" },
  found2d: { en: "client-side JavaScript cannot read the token; a laptop left open does not stay signed in", hi: "क्लाइंट-साइड जावास्क्रिप्ट टोकन नहीं पढ़ सकती; खुला छोड़ा लैपटॉप साइन इन नहीं रहता", mr: "क्लायंट-साइड जावास्क्रिप्ट टोकन वाचू शकत नाही; उघडा राहिलेला लॅपटॉप साइन इन राहत नाही" },
  found3l: { en: "Server-side role checks, not UI hiding", hi: "सर्वर-साइड भूमिका जाँच, यूआई छिपाना नहीं", mr: "सर्व्हर-साइड भूमिका तपासणी, यूआय लपवणे नाही" },
  found3d: { en: "an Evaluator's attempt to start a paid run is rejected with HTTP 403 by the server itself", hi: "एक इवैल्युएटर का सशुल्क रन शुरू करने का प्रयास सर्वर स्वयं HTTP 403 से अस्वीकार करता है", mr: "इव्हॅल्युएटरचा सशुल्क रन सुरू करण्याचा प्रयत्न सर्व्हर स्वतः HTTP 403 ने नाकारतो" },
  found4l: { en: "Append-only traces, secrets never in artefacts", hi: "केवल-जोड़ ट्रेस, रहस्य कभी आर्टिफ़ैक्ट में नहीं", mr: "केवळ-जोडणी ट्रेस, गुपिते कधीही आर्टिफॅक्टमध्ये नाहीत" },
  found4d: { en: "every scored item is written once with its manifest; keys are redacted from logs, traces, cache keys and exports", hi: "हर स्कोर किया आइटम अपने मैनिफ़ेस्ट के साथ एक बार लिखा जाता है; कुंजियाँ लॉग, ट्रेस, कैश कुंजी और निर्यात से हटाई जाती हैं", mr: "प्रत्येक स्कोअर केलेला आयटम त्याच्या मॅनिफेस्टसह एकदाच लिहिला जातो; की लॉग, ट्रेस, कॅश की आणि निर्यातीतून काढल्या जातात" },
  consEyebrow: { en: "THREE CONSOLES, THREE QUESTIONS", hi: "तीन कंसोल, तीन प्रश्न", mr: "तीन कन्सोल, तीन प्रश्न" },
  consTitle: { en: "Same platform, different work", hi: "एक ही प्लेटफ़ॉर्म, अलग काम", mr: "तेच प्लॅटफॉर्म, वेगळे काम" },
  cons1t: { en: "Evaluate", hi: "मूल्यांकन", mr: "मूल्यांकन" },
  cons1d: { en: "Pick the task, see the data format with sample rows, have the profile written and validated, upload a file checked line by line, then run.", hi: "कार्य चुनें, नमूना पंक्तियों सहित डेटा प्रारूप देखें, प्रोफ़ाइल लिखवाएँ और सत्यापित कराएँ, पंक्ति दर पंक्ति जाँची फ़ाइल अपलोड करें, फिर चलाएँ।", mr: "कार्य निवडा, नमुना ओळींसह डेटा स्वरूप पहा, प्रोफाइल लिहून व पडताळून घ्या, ओळीने तपासलेली फाइल अपलोड करा, मग चालवा." },
  cons2t: { en: "Case studies", hi: "केस स्टडी", mr: "केस स्टडी" },
  cons2d: { en: "Five Indian-government use cases in English, Hindi and Hinglish, each with a write-up and the newest live run.", hi: "अंग्रेज़ी, हिंदी और हिंग्लिश में पाँच भारतीय शासकीय उपयोग, हर एक लेख और नवीनतम लाइव रन सहित।", mr: "इंग्रजी, हिंदी आणि हिंग्लिशमध्ये पाच भारतीय शासकीय वापर, प्रत्येक लेख आणि नवीनतम थेट रनसह." },
  cons3t: { en: "Decide", hi: "निर्णय", mr: "निर्णय" },
  cons3d: { en: "Not the highest score: the cheapest model that clears your bar at your daily volume, with the cost of every extra point.", hi: "सबसे ऊँचा स्कोर नहीं: आपकी दैनिक मात्रा पर आपकी सीमा पार करने वाला सबसे सस्ता मॉडल, हर अतिरिक्त अंक की कीमत सहित।", mr: "सर्वाधिक स्कोअर नाही: तुमच्या दैनंदिन प्रमाणात तुमची मर्यादा पार करणारे सर्वात स्वस्त मॉडेल, प्रत्येक अतिरिक्त गुणाच्या किमतीसह." },
  readyTitle: { en: "Ready to find out which model actually clears your bar?", hi: "जानने को तैयार हैं कि कौन सा मॉडल वास्तव में आपकी सीमा पार करता है?", mr: "कोणते मॉडेल खरोखर तुमची मर्यादा पार करते हे जाणून घेण्यास तयार आहात?" },
  readyBody: { en: "Seventeen screens, one trace store, one significance implementation: evaluation, benchmarks, probes and decisions, built for institutions that have to answer for the models they deploy.", hi: "सत्रह स्क्रीन, एक ट्रेस स्टोर, एक महत्व कार्यान्वयन: मूल्यांकन, बेंचमार्क, प्रोब और निर्णय, उन संस्थाओं के लिए जिन्हें अपने मॉडलों का जवाब देना है।", mr: "सतरा स्क्रीन, एक ट्रेस स्टोअर, एक लक्षणीयता अंमलबजावणी: मूल्यांकन, बेंचमार्क, प्रोब आणि निर्णय, ज्यांना त्यांच्या मॉडेलचे उत्तर द्यावे लागते अशा संस्थांसाठी." },
  footerTag: { en: "Evaluation Harness · Institutional LLM evaluation", hi: "इवैल्युएशन हार्नेस · संस्थागत एलएलएम मूल्यांकन", mr: "इव्हॅल्युएशन हार्नेस · संस्थात्मक एलएलएम मूल्यांकन" },
  footerNote: { en: "A working pilot, honest about what is measured and what is not; see", hi: "एक कार्यशील पायलट, जो मापा गया और जो नहीं, उसके बारे में ईमानदार; देखें", mr: "एक कार्यरत पायलट, काय मोजले आणि काय नाही याबद्दल प्रामाणिक; पहा" },
  footerLink: { en: "what's actually real", hi: "क्या वास्तव में वास्तविक है", mr: "प्रत्यक्षात काय खरे आहे" },
  back: { en: "‹ Back", hi: "‹ वापस", mr: "‹ मागे" },
  principle: { en: "Platform principle: cost is negatively weighted, and a non-significant gap is never a winner", hi: "प्लेटफ़ॉर्म सिद्धांत: खर्च ऋणात्मक भार रखता है, और गैर-महत्वपूर्ण अंतर कभी विजेता नहीं", mr: "प्लॅटफॉर्म तत्त्व: खर्च ऋण भार धरतो, आणि गैर-लक्षणीय फरक कधीही विजेता नाही" },
  heroTitle1: { en: "Numbers that", hi: "ऐसे आँकड़े जो", mr: "असे आकडे जे" },
  heroTitleAccent: { en: "hold up to scrutiny.", hi: "जाँच में खरे उतरें।", mr: "छाननीत टिकतात." },
  heroBody: { en: "A single gated entry point for this pilot. Every model sees the same items in the same order, every paid call is metered, and nothing counts as a winner until the paired test says the gap is real.", hi: "इस पायलट के लिए एक ही गेटेड प्रवेश। हर मॉडल वही आइटम उसी क्रम में देखता है, हर सशुल्क कॉल मापी जाती है, और युग्मित परीक्षण द्वारा अंतर वास्तविक बताए जाने तक कुछ भी विजेता नहीं गिना जाता।", mr: "या पायलटसाठी एकच गेटेड प्रवेश. प्रत्येक मॉडेल तेच आयटम त्याच क्रमाने पाहते, प्रत्येक सशुल्क कॉल मोजला जातो, आणि जोडी चाचणी फरक खरा असल्याचे सांगेपर्यंत काहीही विजेता गणले जात नाही." },
  check1: { en: "Role-based access for Assurance Leads and Evaluators, enforced server-side", hi: "एश्योरेंस लीड और इवैल्युएटर के लिए भूमिका-आधारित पहुँच, सर्वर-साइड लागू", mr: "अ‍ॅश्युरन्स लीड आणि इव्हॅल्युएटरसाठी भूमिका-आधारित प्रवेश, सर्व्हर-साइड लागू" },
  check2: { en: "Every run records its manifest: git SHA, dataset and apparatus hashes, pricing version", hi: "हर रन अपना मैनिफ़ेस्ट दर्ज करता है: गिट SHA, डेटासेट और उपकरण हैश, मूल्य संस्करण", mr: "प्रत्येक रन आपला मॅनिफेस्ट नोंदवतो: गिट SHA, डेटासेट आणि उपकरण हॅश, किंमत आवृत्ती" },
  check3: { en: "Traces are append-only; re-scoring writes a derived table, never a mutation", hi: "ट्रेस केवल-जोड़ हैं; पुनः-स्कोरिंग व्युत्पन्न तालिका लिखती है, कभी परिवर्तन नहीं", mr: "ट्रेस केवळ-जोडणी आहेत; पुन्हा-स्कोअरिंग व्युत्पन्न तक्ता लिहिते, कधीही बदल नाही" },
  statInvariantsU: { en: "INVARIANTS ENFORCED", hi: "लागू अपरिवर्तनीय नियम", mr: "लागू अपरिवर्तनीय नियम" },
  statMetered: { en: "SPEND METERED", hi: "खर्च मापा गया", mr: "खर्च मोजला" },
  statCasesU: { en: "CASE STUDIES", hi: "केस स्टडी", mr: "केस स्टडी" },
  badge1: { en: "Paired Tests", hi: "युग्मित परीक्षण", mr: "जोडी चाचण्या" },
  badge2: { en: "Metered Spend", hi: "मापा खर्च", mr: "मोजलेला खर्च" },
  badge3: { en: "Append-Only Traces", hi: "केवल-जोड़ ट्रेस", mr: "केवळ-जोडणी ट्रेस" },
  secureSignIn: { en: "Secure Sign-In", hi: "सुरक्षित साइन-इन", mr: "सुरक्षित साइन-इन" },
  formIntro: { en: "Every field below feeds the attribution on the run manifests, not a production credential store.", hi: "नीचे का हर फ़ील्ड रन मैनिफ़ेस्ट की एट्रिब्यूशन में जाता है, किसी उत्पादन क्रेडेंशियल स्टोर में नहीं।", mr: "खालील प्रत्येक फील्ड रन मॅनिफेस्टच्या श्रेयनिर्देशात जाते, उत्पादन क्रेडेन्शियल स्टोअरमध्ये नाही." },
  yourName: { en: "YOUR NAME", hi: "आपका नाम", mr: "तुमचे नाव" },
  function: { en: "INSTITUTIONAL FUNCTION", hi: "संस्थागत कार्य", mr: "संस्थात्मक कार्य" },
  selectRole: { en: "SELECT A PILOT ROLE TO CONTINUE", hi: "जारी रखने के लिए पायलट भूमिका चुनें", mr: "पुढे जाण्यासाठी पायलट भूमिका निवडा" },
  roleLead: { en: "Assurance Lead", hi: "एश्योरेंस लीड", mr: "अ‍ॅश्युरन्स लीड" },
  roleLeadDesc: { en: "Can start runs that spend the provider key; the only role gated server-side today", hi: "प्रदाता कुंजी खर्च करने वाले रन शुरू कर सकता है; आज सर्वर-साइड गेटेड एकमात्र भूमिका", mr: "पुरवठादार की खर्च करणारे रन सुरू करू शकतो; आज सर्व्हर-साइड गेटेड एकमेव भूमिका" },
  roleEval: { en: "Evaluator", hi: "इवैल्युएटर", mr: "इव्हॅल्युएटर" },
  roleEvalDesc: { en: "Reads reports, case studies and past attempts; starting a paid run needs an Assurance Lead", hi: "रिपोर्ट, केस स्टडी और पिछले प्रयास पढ़ता है; सशुल्क रन शुरू करने के लिए एश्योरेंस लीड चाहिए", mr: "अहवाल, केस स्टडी आणि मागील प्रयत्न वाचतो; सशुल्क रन सुरू करण्यासाठी अ‍ॅश्युरन्स लीड लागतो" },
  password: { en: "PASSWORD", hi: "पासवर्ड", mr: "पासवर्ड" },
  pwPlaceholder: { en: "Shared pilot password", hi: "साझा पायलट पासवर्ड", mr: "सामायिक पायलट पासवर्ड" },
  remember: { en: "Remember my function and role on this device", hi: "इस डिवाइस पर मेरा कार्य और भूमिका याद रखें", mr: "या डिव्हाइसवर माझे कार्य आणि भूमिका लक्षात ठेवा" },
  enterWorkspace: { en: "Enter the workspace", hi: "कार्यक्षेत्र में प्रवेश करें", mr: "कार्यक्षेत्रात प्रवेश करा" },
  signingIn: { en: "Signing in…", hi: "साइन इन हो रहा है…", mr: "साइन इन होत आहे…" },
  noteStrong: { en: "Shared pilot password, not a personal account.", hi: "साझा पायलट पासवर्ड, व्यक्तिगत खाता नहीं।", mr: "सामायिक पायलट पासवर्ड, वैयक्तिक खाते नाही." },
  noteRest: { en: "Name and function are unverified attribution only. Role is enforced server-side. Production would use individual accounts, SSO and MFA.", hi: "नाम और कार्य केवल असत्यापित एट्रिब्यूशन हैं। भूमिका सर्वर-साइड लागू होती है। उत्पादन में व्यक्तिगत खाते, SSO और MFA होंगे।", mr: "नाव आणि कार्य केवळ असत्यापित श्रेयनिर्देश आहेत. भूमिका सर्व्हर-साइड लागू होते. उत्पादनात वैयक्तिक खाती, SSO आणि MFA असतील." },
  attribution: { en: "This session, and every run it starts, is attributed to the name above and written to the run manifests.", hi: "यह सत्र, और इससे शुरू हर रन, ऊपर के नाम को एट्रिब्यूट होता है और रन मैनिफ़ेस्ट में लिखा जाता है।", mr: "हे सत्र, आणि त्यातून सुरू होणारा प्रत्येक रन, वरील नावाला श्रेय दिला जातो आणि रन मॅनिफेस्टमध्ये लिहिला जातो." },
};

const DEPARTMENTS = [
  "Business / Programme Owner", "AI Governance (CDAO/CAIO)", "Technology (CDTO/CIO)",
  "IT Security (CISO)", "Privacy (DPO)", "Risk & Compliance", "Independent Assurance",
  "Board / Governing Authority",
];

const FLOW = ["Prepare", "Preflight", "Run", "Report", "Decide", "Gate", "Save"];

const MODULES = [
  ["Overview", "runs, spend, past attempts"], ["Evaluate", "data format, sample rows, a profile written for you"],
  ["Case studies", "five worked evaluations with live results"], ["Preflight", "validate the profile, estimate the bill including the judge"],
  ["Retrieval", "index a corpus with the pinned embedder"], ["Probes", "adversarial items derived from your own"],
  ["Extraction", "how an answer is read out of a reply"], ["Profile run", "pick models, set a budget, run"],
  ["Benchmark run", "public benchmarks with chance-adjusted accuracy"], ["Profile report", "intervals, paired test, cost per correct answer"],
  ["Benchmark results", "extraction failures kept apart from wrong answers"], ["Decide", "the cheapest model clearing your bar at your volume"],
  ["Gate", "regression gate between two runs"], ["Arena", "pairwise judge with position swap"],
  ["Saved reports", "frozen with caveats"], ["Catalogue", "benchmarks with licence and chance level"],
  ["Connections", "provider keys and model probing"],
];

let lang = "en";
try { lang = LANGS.includes(localStorage.getItem("harness-lang")) ? localStorage.getItem("harness-lang") : "en"; } catch { /* private mode */ }

function t(key) {
  const e = STR[key];
  return e ? (e[lang] || e.en) : key;
}

function applyStrings() {
  document.documentElement.lang = lang;
  for (const node of $$("[data-i18n]")) node.textContent = t(node.dataset.i18n);
  for (const node of $$("[data-i18n-placeholder]")) node.placeholder = t(node.dataset.i18nPlaceholder);
  for (const box of $$(".lang-switch")) {
    box.replaceChildren(...LANGS.map((code) => {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = LANG_LABELS[code];
      b.className = code === lang ? "lang-active" : "";
      b.setAttribute("aria-pressed", String(code === lang));
      b.addEventListener("click", () => { lang = code; try { localStorage.setItem("harness-lang", code); } catch { /* ignore */ } applyStrings(); });
      return b;
    }));
  }
}

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("harness-theme"); } catch { /* private mode */ }
  if (saved) document.documentElement.dataset.theme = saved;
  for (const id of ["#themebtn", "#themebtn2"]) {
    const b = $(id);
    if (!b) continue;
    b.addEventListener("click", () => {
      const cur = document.documentElement.dataset.theme
        || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
      const next = cur === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("harness-theme", next); } catch { /* ignore */ }
    });
  }
}

function showLogin(show) {
  $("#hero-view").hidden = show;
  $("#login-view").hidden = !show;
  if (show) { $("#f-name").focus(); window.scrollTo(0, 0); }
}

function initForm() {
  const dept = $("#f-dept");
  dept.replaceChildren(...DEPARTMENTS.map((d) => { const o = document.createElement("option"); o.textContent = d; o.value = d; return o; }));
  let prefs = {};
  try { prefs = JSON.parse(localStorage.getItem("harness-login-prefs") || "{}") || {}; } catch { /* ignore */ }
  if (prefs.name) $("#f-name").value = prefs.name;
  if (prefs.department && DEPARTMENTS.includes(prefs.department)) dept.value = prefs.department;
  let role = prefs.role === "Assurance Lead" ? "Assurance Lead" : "Evaluator";
  const paint = () => { for (const c of $$(".role-card")) c.classList.toggle("role-card-active", c.dataset.role === role); };
  for (const c of $$(".role-card")) c.addEventListener("click", () => { role = c.dataset.role; paint(); });
  paint();

  $("#pw-toggle").addEventListener("click", () => {
    const pw = $("#f-pw");
    pw.type = pw.type === "password" ? "text" : "password";
  });

  $("#loginform").addEventListener("submit", async (e) => {
    e.preventDefault();
    const err = $("#login-error");
    err.hidden = true;
    const submit = $("#submit");
    submit.disabled = true;
    submit.textContent = t("signingIn");
    const payload = { name: $("#f-name").value, department: dept.value, role, password: $("#f-pw").value };
    try {
      const r = await fetch("/api/auth/login", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
      });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.error || ("HTTP " + r.status));
      try {
        if ($("#f-remember").checked) localStorage.setItem("harness-login-prefs", JSON.stringify({ name: payload.name, department: payload.department, role }));
        else localStorage.removeItem("harness-login-prefs");
      } catch { /* ignore */ }
      // The server now serves the app at "/"; the hash (a deep link such as
      // #evaluate) survives the reload, so the reader lands where they meant to.
      location.reload();
    } catch (ex) {
      err.textContent = ex.message || "Sign-in failed";
      err.hidden = false;
      submit.disabled = false;
      submit.textContent = t("enterWorkspace");
    }
  });
}

function initHero() {
  $("#flow").replaceChildren(...FLOW.flatMap((s, i) => {
    const chip = document.createElement("span"); chip.className = "landing-flow-chip"; chip.textContent = s.toUpperCase();
    const out = [chip];
    if (i < FLOW.length - 1) { const a = document.createElement("span"); a.className = "landing-flow-arrow"; a.textContent = "→"; out.push(a); }
    return out;
  }));
  $("#modules").replaceChildren(...MODULES.map(([title, body], i) => {
    const card = document.createElement("div"); card.className = "landing-module-card";
    const head = document.createElement("div"); head.className = "landing-module-head";
    const n = document.createElement("span"); n.className = "landing-module-n"; n.textContent = String(i + 1);
    const pill = document.createElement("span"); pill.className = "pill built"; pill.textContent = "Built";
    head.append(n, pill);
    const h = document.createElement("h4"); h.textContent = title;
    const p = document.createElement("p"); p.textContent = body;
    card.append(head, h, p);
    return card;
  }));
  for (const id of ["#signin-top", "#signin-hero", "#signin-cta"]) $(id).addEventListener("click", () => showLogin(true));
  $("#back").addEventListener("click", () => showLogin(false));
}

initTheme();
initHero();
initForm();
applyStrings();
// A deep link into the app (any hash) means someone was sent to a screen: go
// straight to the sign-in so one reload lands them there.
if (location.hash && location.hash.length > 1) showLogin(true);
