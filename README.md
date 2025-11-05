# ai-powered-indian-constitution-chatbot-frontend
🧠 AI-Powered Indian Constitution Chatbot / AI-पावर्ड इंडियन कॉन्स्टिट्यूशन चैटबॉट
🧠 AI-Powered Indian Constitution Chatbot
🇮🇳 English Description

Overview:
The AI-Powered Indian Constitution Chatbot is an intelligent legal assistant that helps lawyers, students, and citizens understand the Constitution of India quickly and accurately.
It uses advanced AI and vector search to answer queries about constitutional articles, fundamental rights, duties, amendments, and more — all in natural language.

✨ Tech Stack Used:

Frontend: React, CSS

Backend: Flask (Python)

AI Model: LLaMA 4 (via Grok API)

Vector Database: FAISS (for semantic search and document retrieval)

Data Source: Official Indian Constitution PDF

⚙️ How It Works:

The Constitution PDF is converted into text and stored as embeddings in a FAISS vector database.

When a user asks a question, the system searches the most relevant sections from the FAISS DB.

Those sections are passed to the LLaMA 4 model, which generates an accurate, context-based answer.

The React frontend provides a simple and user-friendly chat interface for interaction.

💡 Problem It Solves:

Helps lawyers, students, and the general public quickly access accurate constitutional information.

Reduces dependency on manual searching through long legal texts.

Enables AI-assisted legal research and quick references for articles, rights, and amendments.

Acts as a 24×7 AI legal assistant for Indian law-related queries.

🇮🇳 हिंदी विवरण

परिचय:
AI-पावर्ड इंडियन कॉन्स्टिट्यूशन चैटबॉट एक बुद्धिमान कानूनी सहायक है जो वकीलों, छात्रों और आम नागरिकों को भारत के संविधान को तेज़ी और सटीकता से समझने में मदद करता है।
यह चैटबॉट AI और वेक्टर सर्च तकनीक का उपयोग करके संविधान से जुड़े प्रश्नों के उत्तर देता है — जैसे अनुच्छेद, मौलिक अधिकार, कर्तव्य और संशोधन आदि।

✨ उपयोग की गई टेक्नोलॉजी:

फ्रंटएंड: React, CSS

बैकएंड: Flask (Python)

AI मॉडल: LLaMA 4 (Grok API के माध्यम से)

वेक्टर डेटाबेस: FAISS (सेमांटिक सर्च और डाक्युमेंट रिट्रीवल के लिए)

डेटा स्रोत: भारत का आधिकारिक संविधान PDF

⚙️ यह कैसे काम करता है:

संविधान PDF को टेक्स्ट में बदलकर FAISS वेक्टर डेटाबेस में सेव किया जाता है।

जब यूज़र कोई प्रश्न पूछता है, सिस्टम सबसे संबंधित सेक्शन खोजता है।

इन सेक्शन को LLaMA 4 मॉडल को भेजा जाता है जो संदर्भ आधारित उत्तर तैयार करता है।

React फ्रंटएंड एक सरल और इंटरएक्टिव चैट इंटरफेस प्रदान करता है।

💡 यह किन समस्याओं को हल करता है:

वकीलों, छात्रों और नागरिकों को संविधान की जानकारी आसानी से मिलती है।

लंबे कानूनी दस्तावेज़ों में मैन्युअल खोज की आवश्यकता नहीं रहती।

AI-सहायता प्राप्त कानूनी रिसर्च को सक्षम बनाता है।

भारतीय संविधान से जुड़े सवालों के लिए यह एक 24×7 AI लीगल असिस्टेंट के रूप में काम करता है।
