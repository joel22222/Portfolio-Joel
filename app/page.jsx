"use client";

import { useState } from 'react';
import { Brain, Github, Linkedin, Mail, ExternalLink, Code, Menu, X, ChevronRight, Target, Award, Sparkles } from 'lucide-react';

export default function AAIPortfolio() {
  const [selectedProject, setSelectedProject] = useState(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // EDIT YOUR PROJECTS HERE
  const projects = [
    {
      id: 1,
      title: 'Real-time Anomaly Detection for Industrial Equipment',
      category: 'Unsupervised Learning',
      tags: ['Isolation Forest', 'PCA', 'Autoencoder', 'Predictive Maintenance'],
      image: 'https://images.unsplash.com/photo-1581092160562-40aa08e78837?w=800&h=600&fit=crop',
      shortDesc: 'Unsupervised anomaly detection pipeline for monitoring industrial equipment using multivariate sensor data',
      problem: 'Industrial plants use many sensors (vibration, temperature, pressure, flow, RPM) to monitor critical rotating equipment. Manual inspection of multivariate sensor feeds is infeasible at scale and real failure labels are rare. The core challenge is detecting unusual patterns and early signs of equipment degradation from streaming sensor data without reliable labeled faults, while balancing false positives and false negatives across noisy, heterogeneous sensor channels.',
      solution: 'Built an unsupervised anomaly-detection pipeline comparing five model families (Isolation Forest, DBSCAN, Autoencoder, One-Class SVM, LOF). Evaluated three scalers (StandardScaler, MinMaxScaler, RobustScaler) to address distributional assumptions and extreme outliers. Used PCA projections to visualize separation between normal and anomalous points. After systematic hyperparameter tuning, selected Fine-Tuned Isolation Forest with RobustScaler for clearest separation and most reliable anomaly signals.',
      implementation: [
        'Ingested time-series sensor data and one-hot encoded categorical features (equipment type, location)',
        'Created derived features including rolling means, deltas, and normalized vibration magnitude',
        'Preserved extreme outliers (potential faults) rather than removing them, using RobustScaler to handle sensitivity',
        'Trained baseline models (Isolation Forest, DBSCAN, Autoencoder, LOF, One-Class SVM) with default parameters',
        'Applied PCA for 2D visualization to guide model selection - plotted inliers vs outliers for each model/scaler combination',
        'Performed manual hyperparameter tuning on Isolation Forest (contamination rate, n_estimators, max_samples) guided by PCA visuals',
        'Built Autoencoder with architecture: Input → Dense 128 → Dense 64 → Bottleneck 16 → Dense 64 → Dense 128 → Output, trained to minimize reconstruction MSE',
        'Evaluated models using Precision, Recall, F1 scores on labeled/synthesized evaluation subsets',
        'Selected Isolation Forest + RobustScaler as final model based on PCA separation and detection metrics'
      ],
      results: [
        'Fine-tuned Isolation Forest produced clearest PCA separation - dense central inlier cluster with distinct scattered outliers',
        'RobustScaler reduced sensitivity to extreme values and provided stable anomaly scores across parameter choices',
        'Autoencoder achieved 0.79 F1 score but showed more overlap in PCA plots compared to Isolation Forest',
        'Early anomaly detection enables targeted inspections, reducing unplanned downtime and maintenance costs while improving plant reliability'
      ],
      technical: 'Built with Python, scikit-learn (IsolationForest, DBSCAN, LOF, OneClassSVM, PCA), PyTorch/Keras for Autoencoder. Used pandas/numpy for data processing, matplotlib/seaborn for visualization. Explored GridSearchCV and RandomizedSearchCV for initial hyperparameter sweeps, followed by manual fine-tuning. Deployment planned via Flask/FastAPI microservice for real-time scoring.',
      challenges: 'Lack of labeled failure data required unsupervised approaches and visual diagnostics (PCA) plus expert-labeled subsets for validation. Traditional outlier removal would erase the fault signal, so preserved extremes using RobustScaler. Different scalers and hyperparameters produced widely varying results - systematic testing with PCA feedback was crucial. Maintained ensemble approach with primary Isolation Forest detector and secondary checks (Autoencoder, LOF) to reduce false positives.',
      github: 'https://github.com/joel22222/AIAM/blob/main/AIAMSUPERFINAL.ipynb',
      year: '2025'
    },
    {
      id: 2,
      title: 'Phishing Email Detection with Poisoned Training Data Defense',
      category: 'ML Security',
      tags: ['Cybersecurity', 'MITRE ATLAS', 'NLP', 'Data Poisoning'],
      image: 'https://images.unsplash.com/photo-1563986768609-322da13575f3?w=800&h=600&fit=crop',
      shortDesc: 'ML pipeline for phishing detection with comprehensive defense against adversarial data poisoning attacks',
      problem: 'Phishing has become one of the most damaging cyber threats, with traditional rule-based filters unable to keep up with rapidly evolving techniques including social engineering, obfuscated URLs, and AI-generated content. The challenge is building an ML model that reliably distinguishes legitimate emails from phishing attempts, even when attackers intentionally corrupt the training data through poisoning attacks - a major vulnerability highlighted in the MITRE ATLAS framework (AML.T0020).',
      solution: 'Built a machine learning pipeline using TF-IDF features and ensemble methods (Voting Classifier combining Logistic Regression, Random Forest, Naive Bayes, SVM) for phishing classification. Simulated 5 poisoning attack strategies (label flipping, backdoor triggers, boundary shifting, data drift, feedback loop injection) to understand vulnerabilities. Developed comprehensive defense framework combining data provenance enforcement, anomaly detection, influence function scanning, and semi-supervised label checking to secure the ML training pipeline.',
      implementation: [
        'Cleaned email text/metadata and applied TF-IDF vectorization (unigrams + bigrams) for feature extraction',
        'Built Voting Classifier with soft voting combining Logistic Regression, Random Forest, Naive Bayes, and SVM for robust predictions',
        'Attempted CNN approach - discovered CNNs fail with TF-IDF due to sparse, non-sequential, non-embedded nature (valuable negative result)',
        'Mapped vulnerabilities using MITRE ATLAS framework (AML.T0020 - Poison Training Data)',
        'Simulated 5 poisoning attacks: label flipping (mislabel phishing as legitimate), backdoor triggers (hidden bypass patterns), decision boundary shifting (borderline samples with wrong labels), data drift poisoning (new trends mislabeled), and feedback loop injection',
        'Implemented defense mechanisms: data provenance tracking, outlier/anomaly detection on training batches, influence function scanning to identify high-impact poisoned samples',
        'Created trusted subset divergence testing - trained parallel model on verified clean data to detect poisoning-induced drift',
        'Applied semi-supervised label checking using confidence thresholds and clustering to catch label flipping',
        'Documented complete attack-defense analysis providing blueprint for robust cybersecurity ML systems'
      ],
      results: [
        'Voting Classifier achieved strong baseline performance under clean training conditions',
        'Label flipping attacks decreased detection accuracy dramatically even with small poisoning percentages',
        'Backdoor poisoning successfully caused specific malicious patterns to completely bypass detection',
        'Decision boundary shifting degraded ensemble reliability by injecting misleading borderline samples',
        'Defense framework successfully detected and mitigated poisoning attempts through multi-layered approach combining provenance, anomaly detection, and influence analysis'
      ],
      technical: 'Built with Python, scikit-learn (Logistic Regression, Naive Bayes, SVM, Random Forest, Voting Classifier), TF-IDF Vectorizer for NLP feature extraction, pandas/NumPy for data processing. Used MITRE ATLAS Framework for attack/defense mapping. Jupyter Notebook for experimentation with matplotlib/seaborn for visualization. Influence function analysis for detecting high-impact poisoned samples.',
      challenges: 'CNN architecture failed with TF-IDF due to incompatible input format (sparse, orderless vs sequential embeddings required) - documented as learning experience and switched to classical ML. Training data manipulation can alter model behavior without detection - simulated 5 explicit poisoning scenarios to understand failure modes. Model sensitivity to label noise - even small label flips drastically changed ensemble outputs, requiring semi-supervised label checking. Backdoor trigger poisoning creates hidden bypasses - mitigated with anomaly detection and influence-function scanning.',
      github: 'https://github.com/joel22222/AICS/blob/main/joelaicslegit%20(2).ipynb',
      year: '2025'
    },
    {
      id: 3,
      title: 'RAG-Powered Complaint Classification & Response Chatbot',
      category: 'NLP & RAG',
      tags: ['Rasa', 'LangChain', 'FastAPI', 'Customer Service AI'],
      image: 'https://images.unsplash.com/photo-1553877522-43269d4ea984?w=800&h=600&fit=crop',
      shortDesc: 'Intelligent chatbot combining SVM classification with RAG for automated customer complaint handling and response generation',
      problem: 'Financial institutions receive thousands of daily customer complaints about billing errors, unauthorized transactions, double charges, and service issues. Manual complaint processing is slow, resource-intensive, prone to misclassification, and causes delays that negatively impact customer satisfaction. Many cases remain unresolved, miscategorized, or overlooked, while companies struggle to identify complaint trends and prioritize urgent cases effectively.',
      solution: 'Developed an NLP-powered Retrieval-Augmented Generation (RAG) chatbot that automatically classifies complaints using SVM, retrieves context-specific answers from company policy documents, and provides immediate responses. Built using Rasa for conversation flow, FastAPI for API layer, LangChain + Ollama for LLM inference, and MySQL for complaint storage. The system automates early-stage complaint handling, enabling faster responses, better case prioritization, and trend analysis for institutions.',
      implementation: [
        'Performed exploratory data analysis on complaint dataset - analyzed label distribution, cleaned text (removed punctuation, URLs, stopwords), handled missing values, applied one-hot encoding',
        'Applied TF-IDF vectorization to convert complaint messages into numerical representations for ML classification',
        'Trained Support Vector Machine (SVM) classifier to categorize complaints into credit_card vs billing categories',
        'Evaluated model using accuracy, confusion matrix, precision, and recall metrics',
        'Built Rasa chatbot with hard-coded conversation flow - implemented intents (greet, provide_complaint_description, thank_you), defined slots and stories',
        'Configured chatbot to store classified complaints in MySQL database',
        'Created standalone RAG prototype using LangChain + Ollama with PDF document containing credit card & billing FAQ as retrieval source',
        'Integrated systems via FastAPI endpoint connecting Rasa → FastAPI → LangChain → Ollama',
        'Deployed unified chatbot enabling users to report complaints, receive automatic classification, and get RAG-generated answers grounded in company policies'
      ],
      results: [
        'SVM classifier achieved strong performance correctly identifying majority of credit card vs billing complaints',
        'Reduced manual classification workload for support teams and enabled priority-based case routing',
        'RAG system provided detailed, accurate answers instantly using reliable company documents',
        'Customers received immediate guidance without waiting for human agents, increasing satisfaction',
        'System enables complaint trend analysis for institutions to identify recurring issues and improve services'
      ],
      technical: 'Built with Python, scikit-learn (SVM, TF-IDF), LangChain for RAG orchestration, Ollama for LLM inference, Rasa for chatbot framework (intents, stories, slots). FastAPI for backend API layer, MySQL for complaint storage, PDF knowledge base with vector retrieval. Development in VS Code and Jupyter Notebook, API testing with Postman.',
      challenges: 'Limited intent classification (only credit_card and billing) with typo failures - planned expansion to fuzzy matching and semantic intent detection using embeddings. Slow response latency from multiple service hops (FastAPI → LangChain → Ollama) - mitigated with caching, local embedding databases, async FastAPI endpoints, and faster LLM models. No cross-session conversation tracking - implementing user login system for historical complaint tracking and context-aware responses. Single PDF knowledge base limits response variety - expanding to multiple policy documents covering mortgages, loans, digital banking, and insurance.',
      year: '2024'
    },
    {
      id: 4,
      title: '3D Avatar Chatbot for Customer Query Resolution',
      category: '3D AI & Voice',
      tags: ['React Three Fiber', 'RAG', 'Voice AI', 'FastAPI'],
      image: 'https://images.unsplash.com/photo-1535378917042-10a22c95931a?w=800&h=600&fit=crop',
      shortDesc: 'Interactive 3D AI avatar with voice interaction for automated customer service in restaurants and hospitality',
      problem: 'Restaurants and hospitality environments face long queues, insufficient staff during peak hours, and customers needing repetitive information (directions, menu details, allergy info, FAQs). Staff cannot efficiently handle these queries, resulting in slow service, frustrated visitors, increased workload, and poor customer experience. The core challenge: how to provide scalable, automated, human-like customer assistance without increasing staffing costs.',
      solution: 'Developed a 3D AI-powered avatar chatbot with real-time voice interaction and visual animation for restaurant/academy environments. Users can ask menu questions, request opening hours/allergy info, get building directions, and engage in small talk using voice or on-screen keyboard. Built with React Three Fiber for 3D rendering, ReadyPlayerMe for character creation, Mixamo for animations, ElevenLabs + Google Speech API for voice, ChromaDB RAG backend for knowledge retrieval, and FastAPI for communication.',
      implementation: [
        'Created 3D model using ReadyPlayerMe, converted and cleaned in Blender, imported into React Three Fiber',
        'Applied Mixamo animations (idle, talking, listening, greeting) and built state machine for smooth conversation transitions',
        'Built voice interaction pipeline: frontend microphone → React Speech Recognition → backend RAG → ElevenLabs TTS synthesis',
        'Integrated Google Speech API as fallback option for speech recognition',
        'Synced avatar mouth animation with audio playback for natural lip-sync effect',
        'Created restaurant knowledge base (menu items, building directions, FAQs) embedded using ChromaDB vector database',
        'Implemented RAG pipeline with custom prompt templates ensuring factual, context-aware responses via FastAPI REST API',
        'Designed UI with floating chat button, on-screen keyboard for noisy environments, map/directions button, built with React + TailwindCSS',
        'Debugged integration issues (CORS, port mismatches) using Postman, consolidated onto single machine, conducted user tests with classmates and staff'
      ],
      realworld: 'Developed and deployed for Temasek Culinary Academy\'s Top Table Restaurant to serve as an intelligent virtual concierge for diners. The 3D avatar chatbot handles customer inquiries about menu items, dietary restrictions, allergen information, reservation details, operating hours, and facility directions. Guests can interact naturally through voice or keyboard, receiving instant responses about ingredient details, dish recommendations, and restaurant information without requiring staff intervention. The system is currently operational at the restaurant, providing 24/7 customer assistance and significantly reducing staff workload during peak dining hours while enhancing the overall dining experience.',
      results: [
        'Fully functional 3D interactive chatbot with natural speech and animated responses including head movement, talking animation, and idle loops',
        'Smooth multi-modal interaction supporting both voice and keyboard input',
        'Reduced staff load during peak hours with consistent, instant replies acting as virtual concierge',
        'User feedback showed avatar felt friendly and engaging with intuitive, futuristic interface much faster than waiting for staff',
        'Easily extendable system ready for additional knowledge bases (menus, events, room locations)'
      ],
      technical: 'Frontend: React, React Three Fiber, Three.js, TailwindCSS, Mixamo animations, React Speech Recognition. Voice: ElevenLabs Text-to-Speech, Google Speech-to-Text. Backend: FastAPI, ChromaDB vector database, custom RAG pipeline, Python. Tools: Blender for model optimization, ReadyPlayerMe for character creation, Postman for API testing, Vite for development.',
      challenges: 'Mixamo animations failed due to 3D model structure mismatches - used Blender to retarget and adjust armature for consistent GLB structure. Hit ElevenLabs voice API limits days before submission - integrated Google Speech API as backup. Frontend-backend integration had CORS and port mismatch issues - combined work onto one machine and debugged with Postman. Steep learning curve with React, React Three Fiber, and Mixamo - learned through tutorials and multiple component rebuilds. 3D model + audio processing slowed on low-end devices - optimized scene lighting, reduced polygon count, limited unnecessary animation updates.',
      year: '2025'
    },
    {
      id: 5,
      title: 'Deep Learning Image Classification with CNN & Transfer Learning',
      category: 'Deep Learning',
      tags: ['CNN', 'Transfer Learning', 'MobileNetV2', 'Android Deployment'],
      image: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=800&h=600&fit=crop',
      shortDesc: 'End-to-end deep learning pipeline comparing multiple CNN architectures, achieving 99% accuracy with fine-tuned MobileNetV2 and successful Android deployment',
      problem: 'Training image classification models from scratch often leads to overfitting due to limited dataset variation, long training times, difficulty achieving stable validation accuracy, and large parameter counts that do not generalize well. Real-world datasets contain mixed image sizes, require minimal cleaning but limited augmentation, and have imbalanced difficulty levels per class. The main challenge: build a high-accuracy image classification model, compare multiple deep learning architectures, fix overfitting, and successfully deploy the final model to Android.',
      solution: 'Developed a full deep learning pipeline including preprocessing, resizing, and augmenting images, a baseline CNN model to understand initial behavior, four transfer learning models (MobileNetV2, ResNet50, VGG16, EfficientNetB0), fine-tuning MobileNetV2 which performed best, exporting the final model to TensorFlow Lite, and deploying the model fully into Android Studio with successful on-device inference. The end result is a mobile application capable of running image classification in real time.',
      implementation: [
        'Loaded dataset directly from ZIP (5000 images per class), resized to 224×224 pixels, applied augmentation (rotations, flips, zoom, shifts) with 80/20 train-validation split',
        'Built baseline CNN with ReLU activation, Adam optimizer (lr=0.001), identified 25M+ parameters causing overfitting risk',
        'Added dropout layers and ReduceLROnPlateau to baseline CNN - training accuracy reached 92% but validation remained unstable at 87% with increasing loss',
        'Implemented transfer learning with MobileNetV2 achieving 94% validation accuracy with smooth curves and no overfitting',
        'Tested ResNet50 (moderate accuracy with slight overfitting), VGG16 (93.57% accuracy with minor fluctuation), EfficientNetB0 (failed at 33% - random guessing)',
        'Fine-tuned MobileNetV2 by enabling training on entire base model for 20 epochs with batch size 32, ReduceLROnPlateau, and EarlyStopping (patience=5)',
        'Achieved final validation accuracy of 99.10% at epoch 18 with strong generalization and stable learning curves',
        'Converted fine-tuned MobileNetV2 to TensorFlow Lite (.tflite) format for mobile deployment',
        'Integrated model into Android Studio with Java/Kotlin inference code, implemented image input pipeline with preprocessing and model prediction',
        'Successfully deployed fully working Android app with fast, accurate on-device inference running smoothly on mobile CPU'
      ],
      results: [
        'Completely eliminated overfitting - MobileNetV2 achieved stable validation curves after fine-tuning with no fluctuation',
        'Achieved 99.10% validation accuracy with fine-tuned model, demonstrating excellent generalization on unseen data',
        'Successfully deployed real mobile application on Android with practical real-world performance and efficient on-device inference',
        'Gained comprehensive understanding of architecture strengths/weaknesses through systematic comparison of CNN, MobileNet, ResNet, VGG, and EfficientNet'
      ],
      technical: 'Built with TensorFlow/Keras, transfer learning architectures (MobileNetV2, VGG16, ResNet50, EfficientNetB0), custom CNN baseline. Data augmentation and image resizing for preprocessing with 80/20 train-validation split. Training optimizations: Adam optimizer, dropout layers, ReduceLROnPlateau, EarlyStopping. Deployment: TensorFlow Lite conversion, Android Studio integration with Kotlin/Java inference API for on-device predictions.',
      challenges: 'Baseline CNN suffered severe overfitting - switched to transfer learning models which immediately fixed the issue. Large dense layer (300k→84 features) caused 25M+ parameter count - avoided by adopting compact, efficient transfer learning architectures. EfficientNetB0 completely failed with 33% accuracy (random guessing) - removed from selection and focused on higher-performing models. Fine-tuning stability required careful LR scheduling and early stopping for huge improvement. Android deployment required TFLite conversion and careful integration - successfully completed with working end-to-end inference pipeline.',
      year: '2024'
    },
    {
      id: 6,
      title: 'Ethical Analysis of AI Misuse in Facial Recognition',
      category: 'AI Ethics',
      tags: ['Bias in AI', 'Facial Recognition', 'Responsible AI', 'Dataset Bias'],
      image: 'https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=800&h=600&fit=crop',
      shortDesc: 'Critical examination of the 2015 Google Photos incident analyzing how biased training data led to racist misclassification and proposing frameworks for responsible AI development',
      problem: 'Artificial Intelligence is increasingly integrated into daily life, but unethical AI deployment can cause significant harm. Facial recognition carries major ethical risks including racial and ethnic misidentification, cultural insensitivity, privacy violations, lack of accountability when errors occur, and bias embedded in training datasets. In 2015, Google Photos mistakenly labeled two Black individuals as "gorillas," demonstrating how AI can perpetuate racist stereotypes due to biased training data. The core issue: AI systems trained on unbalanced, non-diverse datasets can produce discriminatory, offensive outputs — and companies often lack accountability, transparency, and error-handling mechanisms to prevent such harm.',
      solution: 'This project critically examines the Google Photos mislabeling incident, how facial recognition is used across various sectors, the ethical issues that arise when AI systems fail, contributing factors behind AI bias and inaccurate classifications, and practical, realistic solutions that organizations can implement. The focus is on understanding why AI bias happens, what structures allow it to happen, and how companies can prevent such failures through proper dataset curation, cultural awareness, auditing, transparency, and accountability frameworks.',
      implementation: [
        'Analyzed the Google Photos incident: AI image labeling system mislabeled Black couple as "gorillas," Google removed label entirely (quick fix, not real solution), identified root cause as biased non-diverse training data lacking representation of darker skin tones',
        'Examined facial recognition use cases across sectors: security/law enforcement (suspect identification, public monitoring), healthcare (patient ID, health condition detection), authentication (smartphones, banking), retail/marketing (customer behavior), social media (AR filters, tagging), attendance/access control',
        'Identified five major ethical issues: racial misidentification due to biased data, cultural/historical insensitivity (term "gorilla" evokes racism and human zoos), lack of accountability when AI fails, lack of transparency in decision-making, insufficient error handling with no pre-checks for harmful labels',
        'Connected incident to painful historical context: Black people exhibited in "human zoos" in 1800s, racial slurs comparing Black individuals to animals, AI failures unintentionally reviving racist narratives',
        'Analyzed technical factors: dataset bias (underrepresentation of darker skin tones, imbalanced categories, lack of diversity in benchmarks), black box problem in deep learning CNNs (no explainable reasoning), difficulty tracing harmful decisions',
        'Proposed ethical and cultural review teams including sociologists, historians, ethicists, and diversity experts to prevent offensive outputs and encourage inclusive design',
        'Recommended diverse training data from global datasets representing different skin tones, ethnic backgrounds, and lighting conditions with bias mitigation techniques',
        'Suggested interpretable AI models using SHAP (SHapley Additive Explanations) to explain which features influenced decisions and detect embedded bias',
        'Designed accountability frameworks requiring clear documentation, explainable decisions, review boards, and ethical approval workflows',
        'Developed stronger error handling systems with offensive-term filters, human verification for high-risk outputs, and continuous monitoring'
      ],
      results: [
        'Documented negative impacts: emotional and psychological harm to affected individuals, loss of trust in AI systems, public backlash and reputational damage, reinforcement of harmful racial stereotypes, increased scrutiny on AI technologies, potential legal and ethical consequences',
        'Raised awareness about bias in AI systems and highlighted critical importance of diverse, representative datasets for fair AI development',
        'Provided actionable strategies to improve fairness including cultural review processes, bias mitigation techniques, explainability tools, and accountability frameworks',
        'Demonstrated that responsible AI is not just about better accuracy but about respecting human dignity, preventing harm, and ensuring technology benefits everyone equally'
      ],
      technical: 'Analysis framework covering AI ethics principles, dataset bias mechanisms (underrepresentation, imbalanced categories, lack of diversity), black box problem in deep learning CNNs, AI explainability tools (SHAP - SHapley Additive Explanations), error handling and model auditing (pre-deployment testing, continuous dataset reviews, human-in-the-loop validation). Technical concepts: facial recognition systems, convolutional neural networks, image classification, bias detection and mitigation strategies.',
      challenges: 'Racial and ethnic misidentification rooted in biased datasets with lack of demographic diversity. Cultural sensitivity issues - developers lacked awareness of historical racism with no internal cultural review process. Lack of accountability across multiple departments with no single responsibility point and no regulatory frameworks to enforce consequences. Lack of transparency in AI decision-making processes with limited documentation and no public explanation from Google initially. Weak error handling with no safety filters, no mechanism to block offensive labels, and AI deployed too early without sufficient testing.',
      report: 'https://docs.google.com/document/d/1WWTVatPPRt0iBYgbpVn7PwzVDEeZej_D/edit',
      year: '2024'
    },
    {
      id: 7,
      title: '3D Avatar Chatbot for Kiosk & Virtual Concierge',
      category: '3D AI & Voice',
      tags: ['3D Avatar', 'Real-Time AI', 'GPT-5.1 Nano', 'Speech Interaction'],
      image: 'https://images.unsplash.com/photo-1535378917042-10a22c95931a?w=800&h=600&fit=crop',
      shortDesc: 'Voice-enabled 3D avatar virtual concierge with real-time AI conversation, validated by SAFRA Tampines Club Manager for real-world deployment',
      problem: 'Traditional kiosk systems rely on static buttons, rigid menus, and outdated touchscreens that cannot hold conversations, answer dynamic questions, confuse first-time users, lack accessibility for elderly or visually impaired visitors, and provide no personality, emotion, or natural interaction. Modern facilities like malls, gyms, and community hubs need a more intuitive, human-like, interactive solution that can greet visitors, answer questions, guide people around the venue, and provide real-time support using natural conversation.',
      solution: 'Designed and built a 3D Avatar Chatbot that functions as a virtual concierge for kiosk-style environments, delivering a futuristic, human-like experience through a fully rigged 3D avatar, real-time AI conversation, voice input with speech output, facial animation with lip-sync, map display with on-screen directions, idle detection with ambient animations, and full kiosk-mode web deployment. Users interact with the voice-enabled 3D assistant as if speaking to a real concierge.',
      implementation: [
        'Built frontend with React, React Three Fiber, @react-three/drei, custom GLB avatar with full facial rig and blendshapes, and interactive UI panels (maps, directions, shortcuts)',
        'Developed Node.js Express backend server with GPT-5.1 Nano as lightweight AI engine, websocket-style messaging for real-time responses, custom prompt framework with conversation memory',
        'Implemented speech system: microphone input → browser transcription → backend processing → GPT-5.1 Nano real-time reply → ElevenLabs TTS natural speech → avatar lip-sync synced to waveform amplitude and phoneme timing',
        'Created smooth interaction pipeline: user talks → audio transcribed to text → backend receives transcript → AI generates reply → speech conversion → avatar plays talking animation with lip-sync → UI panels update',
        'Designed kiosk mode features: auto full-screen interface, idle animation when user silent, map directions panel ("Show me where the gym is"), FAQ buttons, multi-modal interaction (voice + UI)',
        'Implemented lip-sync accuracy using amplitude-based phoneme mapping with smooth interpolation for natural mouth movements',
        'Optimized 3D performance for lower-end hardware through texture optimization, animation compression, and reduced draw calls',
        'Prepared lightweight web deployment optimized for kiosk mode, compatible with touchscreen kiosks, tablets, and displays'
      ],
      realworld: 'Contacted the Club Manager of SAFRA Tampines to demonstrate the 3D avatar kiosk concept for their facility. The manager described the system as "Very intriguing" and stated "We would definitely use something like this," expressing strong interest for member assistance, facility navigation, automated 24/7 reception, and reducing staff workload. While adoption was prevented by government tender requirements (SAFRA must conduct official procurement processes to ensure vendor fairness), this validation from a real organization proves the project has genuine commercial potential and industry relevance.',
      results: [
        'Fully functional 3D voice-enabled avatar with complete animation, lip-sync, and natural conversational flow powered by real-time AI',
        'Fast inference with GPT-5.1 Nano optimized for kiosk interactions, providing immediate responses for seamless user experience',
        'Strong real-world validation: SAFRA Club Manager expressed genuine interest and willingness to use the system, proving industry relevance and commercial potential',
        'Scalable browser-based solution deployable on touchscreen kiosks, tablets, or displays without requiring specialized hardware or installation'
      ],
      technical: 'AI & Backend: GPT-5.1 Nano, Node.js (Express), real-time response handling, custom conversation pipeline with memory. Frontend & 3D: React, React Three Fiber, Drei, GLB models with blendshapes and animations, interactive UI components. Speech: ElevenLabs Text-to-Speech, Browser Speech-to-Text with microphone input. Deployment: Web-based kiosk mode, lightweight hosting optimized for real-time 3D rendering and AI inference.',
      challenges: 'Lip-sync accuracy challenge - making avatar mouth match speech naturally, solved with amplitude-based phoneme mapping and smooth interpolation. Maintaining smooth 3D performance on lower-end kiosk hardware - optimized textures, compressed animations, and reduced draw calls. Real-time AI responsiveness critical for kiosk interfaces - chose GPT-5.1 Nano for extremely low-latency inference. Real-world deployment complexity with SAFRA requiring official government tendering process - documented requirements and prepared for potential future tender opportunity.',
      demo: 'https://youtu.be/Rfrp9hSqC_E',
      year: '2025'
    },
    {
      id: 8,
      title: 'RPA Automation for Booking Confirmation System',
      category: 'Robotic Process Automation',
      tags: ['UiPath', 'Google Workspace', 'Email Automation', 'Excel Automation'],
      image: 'https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?w=800&h=600&fit=crop',
      shortDesc: 'End-to-end UiPath RPA bot automating booking confirmation workflow from Google Forms to customer email delivery with 24/7 operation',
      problem: 'Administrators were manually handling booking submissions, which involved reading new booking entries from Google Forms, copying details into spreadsheets, creating text files with booking information, sending confirmation emails to customers, and handling invalid email inputs manually. These tasks are time-consuming, repetitive, prone to human error, stressful when requests accumulate overnight, and not scalable for high booking volumes. The goal: automate the entire booking confirmation workflow using RPA so the system operates 24/7 without errors, delays, or manual work.',
      solution: 'Designed and built a UiPath RPA bot that fully automates reading new bookings from Google Forms, syncing real-time data from Google Sheets, converting entries into structured Excel data, looping through each booking, generating a detailed text file for each customer, automatically sending emails with the attached text file, and validating email addresses before sending. This solution eliminates manual effort and ensures consistent, professional communication with zero human intervention required.',
      implementation: [
        'User submits booking details via Google Forms, data instantly stored in Google Sheets for real-time access',
        'Connected UiPath bot to Google Workspace using Google Workspace Activities package, extracting real-time data from Google Sheets automatically',
        'Transferred Google Sheets rows to Excel file for structured looping and data processing using Excel Activities',
        'Implemented row-by-row iteration through Excel: bot extracts customer details, creates new text file, appends and formats content using Word Activities for consistent professional formatting',
        'Automated email delivery using UiPath Mail Activities: attaches generated text file, sends to customer email automatically with professional formatting',
        'Built email validation system: if user enters invalid email, bot displays message box alert, blocks email sending, and highlights invalid entry to ensure data accuracy and prevent failures',
        'Integrated Google Workspace API connection to enable direct UiPath-to-Google Sheets authentication and real-time data fetching',
        'Configured 24/7 operation mode allowing bot to process bookings continuously even when staff are offline, handling overnight accumulations automatically'
      ],
      results: [
        '100% automation of entire booking confirmation workflow - no manual reading, copying, or sending emails needed, eliminating all repetitive administrative tasks',
        'Zero human error with immediate flagging of incorrect emails, preventing delivery failures and ensuring data accuracy throughout the process',
        '24/7 continuous operation allowing bot to work when staff are offline, providing instant customer confirmations regardless of time zone or business hours',
        'Significantly reduced staff workload by eliminating tedious repetitive tasks, allowing team to focus on higher-value activities while maintaining higher professionalism through consistent text file and email formatting'
      ],
      technical: 'Built with UiPath Studio using four core activity packages: Google Workspace Activities (connects to Google Sheets, retrieves booking data in real-time), Excel Activities (reads rows, loops through entries, converts data to structured format), Word Activities (appends and formats text files with professional layout), and Mail Activities (sends emails with attachments for automated customer communication). Workflow pipeline: Google Forms → Google Sheets → UiPath data extraction → Excel conversion → text file generation → email validation → automated delivery.',
      challenges: 'Invalid email inputs causing failed deliveries - solved by adding validation that alerts user and blocks email sending. Linking UiPath with Google Sheets requiring authentication - implemented Google Workspace API connection for secure access. Ensuring proper file formatting to avoid messy raw text - used Word Activities for clean, professional formatting. Handling multiple customers efficiently at scale - implemented Excel Activities with row-by-row iteration for reliable bulk processing.',
      year: '2024'
    }
  ];

  const skills = [
    { category: 'Machine Learning', items: ['PyTorch', 'TensorFlow', 'Scikit-learn', 'XGBoost'] },
    { category: 'Deep Learning', items: ['CNN', 'Transformers', 'GANs'] },
    { category: 'NLP & LLMs', items: [ 'GPT', 'LangChain', 'RAG'] },
    { category: 'Computer Vision', items: ['OpenCV', 'Object Detection'] },
    { category: 'MLOps', items: ['Docker', 'MLflow', 'CI/CD'] },
    { category: 'Languages', items: ['Python', 'SQL', 'JavaScript', 'C++'] }
  ];

  const ProjectModal = ({ project, onClose }) => (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 overflow-y-auto">
      <div className="min-h-screen px-4 py-8">
        <div className="max-w-5xl mx-auto bg-white rounded-xl shadow-2xl">
          <div className="relative h-72 overflow-hidden rounded-t-xl">
            <img src={project.image} alt={project.title} className="w-full h-full object-cover" />
            <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
            <button
              onClick={onClose}
              className="absolute top-4 right-4 bg-white/10 backdrop-blur-md p-2 rounded-full hover:bg-white/20 transition"
            >
              <X className="text-white" size={20} />
            </button>
            <div className="absolute bottom-6 left-6 right-6">
              <div className="flex items-center gap-2 mb-2">
                <span className="bg-slate-900/80 backdrop-blur-sm px-3 py-1 rounded-md text-white text-sm font-medium">
                  {project.category}
                </span>
                <span className="bg-white/20 backdrop-blur-sm px-3 py-1 rounded-md text-white text-sm">
                  {project.year}
                </span>
              </div>
              <h2 className="text-3xl font-bold text-white mb-2">{project.title}</h2>
              <div className="flex flex-wrap gap-2">
                {project.tags.map((tag, idx) => (
                  <span key={idx} className="bg-white/20 backdrop-blur-sm px-2 py-1 rounded text-white text-xs">
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className="p-8 space-y-6">
            <div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Problem Statement</h3>
              <p className="text-gray-700 leading-relaxed">{project.problem}</p>
            </div>

            <div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Solution Overview</h3>
              <p className="text-gray-700 leading-relaxed">{project.solution}</p>
            </div>

            <div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Implementation Details</h3>
              <ul className="space-y-2">
                {project.implementation.map((item, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <ChevronRight className="text-slate-600 flex-shrink-0 mt-0.5" size={18} />
                    <span className="text-gray-700 text-sm">{item}</span>
                  </li>
                ))}
              </ul>
            </div>

            {project.realworld && (
              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Real-World Application</h3>
                <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded-r-lg">
                  <p className="text-gray-700 leading-relaxed text-sm">{project.realworld}</p>
                </div>
              </div>
            )}

            <div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Key Results & Impact</h3>
              <div className="space-y-2">
                {project.results.map((result, idx) => (
                  <div key={idx} className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                    <p className="text-gray-800 text-sm">{result}</p>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Technical Stack</h3>
              <p className="text-gray-700 leading-relaxed text-sm">{project.technical}</p>
            </div>

            <div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Challenges & Solutions</h3>
              <p className="text-gray-700 leading-relaxed text-sm">{project.challenges}</p>
            </div>

            <div className="flex gap-3 pt-4">
              {project.github && (
                <a
                  href={project.github}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 bg-slate-900 text-white px-5 py-2.5 rounded-lg text-sm font-medium hover:bg-slate-800 transition"
                >
                  <Github size={18} />
                  View Code
                </a>
              )}
              {project.demo && (
                <a
                  href={project.demo}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 border-2 border-slate-300 text-slate-700 px-5 py-2.5 rounded-lg text-sm font-medium hover:bg-slate-50 transition"
                >
                  <ExternalLink size={18} />
                  Live Demo
                </a>
              )}
              {project.report && (
                <a
                  href={project.report}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 border-2 border-blue-500 text-blue-600 px-5 py-2.5 rounded-lg text-sm font-medium hover:bg-blue-50 transition"
                >
                  <ExternalLink size={18} />
                  Full Report
                </a>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-white">
      {selectedProject && <ProjectModal project={selectedProject} onClose={() => setSelectedProject(null)} />}

      {/* Navigation */}
      <nav className="bg-white border-b border-gray-200 sticky top-0 z-40">
        <div className="max-w-6xl mx-auto px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-slate-900 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">JT</span>
              </div>
              <span className="font-semibold text-gray-900">Joel Tan</span>
            </div>
            
            <div className="hidden md:flex items-center gap-8">
              <a href="#projects" className="text-gray-600 hover:text-gray-900 text-sm font-medium transition">Projects</a>
              <a href="#skills" className="text-gray-600 hover:text-gray-900 text-sm font-medium transition">Skills</a>
              <a href="#about" className="text-gray-600 hover:text-gray-900 text-sm font-medium transition">About</a>
              <a
                href="#contact"
                className="bg-slate-900 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-slate-800 transition"
              >
                Contact
              </a>
            </div>

            <button 
              className="md:hidden"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>

          {mobileMenuOpen && (
            <div className="md:hidden py-4 space-y-3 border-t border-gray-200">
              <a href="#projects" className="block text-gray-600 hover:text-gray-900 text-sm font-medium">Projects</a>
              <a href="#skills" className="block text-gray-600 hover:text-gray-900 text-sm font-medium">Skills</a>
              <a href="#about" className="block text-gray-600 hover:text-gray-900 text-sm font-medium">About</a>
              <a href="#contact" className="block text-gray-600 hover:text-gray-900 text-sm font-medium">Contact</a>
            </div>
          )}
        </div>
      </nav>

      {/* Hero Section */}
      <div className="max-w-6xl mx-auto px-6 lg:px-8 py-24">
        <div className="max-w-3xl">
          <div className="inline-flex items-center gap-1.5 text-slate-600 text-sm mb-6">
            <Sparkles size={16} />
            <span>Available for collaboration</span>
          </div>
          
          <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-4 tracking-tight">
            Joel Tan
          </h1>
          
          <p className="text-xl text-gray-600 mb-8">
            Aspiring AI/ML Engineer building practical solutions in machine learning, NLP, and computer vision.
          </p>
          
          <p className="text-base text-gray-600 mb-10 leading-relaxed">
            I'm passionate about using AI to solve real-world problems. My work focuses on creating systems 
            that are not just technically sound, but genuinely useful. Currently applying to SIT's Applied AI 
            program to deepen my expertise and contribute to the next generation of AI applications.
          </p>

          <div className="flex flex-wrap gap-3 mb-12">
            <a
              href="#projects"
              className="bg-slate-900 text-white px-6 py-3 rounded-lg font-medium hover:bg-slate-800 transition"
            >
              View My Work
            </a>
            <a 
              href="#contact"
              className="border border-gray-300 text-gray-700 px-6 py-3 rounded-lg font-medium hover:border-gray-400 hover:bg-gray-50 transition"
            >
              Get in Touch
            </a>
          </div>

          <div className="flex flex-wrap gap-6 text-sm text-gray-500">
            <div className="flex items-center gap-2">
              <div className="w-1 h-1 bg-gray-400 rounded-full"></div>
              <span>8+ ML/AI Projects</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-1 h-1 bg-gray-400 rounded-full"></div>
              <span>Specialized in NLP & Computer Vision</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-1 h-1 bg-gray-400 rounded-full"></div>
              <span>SIT AAI Applicant</span>
            </div>
          </div>
        </div>
      </div>

      {/* Projects Section */}
      <div id="projects" className="bg-slate-50 py-20">
        <div className="max-w-6xl mx-auto px-6 lg:px-8">
          <div className="mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-3">Featured Projects</h2>
            <p className="text-gray-600 max-w-2xl">
              A collection of my AI/ML work, from anomaly detection to conversational AI. 
              Click on any project to see the full technical details.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {projects.map((project) => (
              <div
                key={project.id}
                onClick={() => setSelectedProject(project)}
                className="bg-white rounded-lg overflow-hidden hover:shadow-lg transition cursor-pointer group border border-gray-200"
              >
                <div className="relative overflow-hidden h-48">
                  <img 
                    src={project.image} 
                    alt={project.title}
                    className="w-full h-full object-cover group-hover:scale-105 transition duration-300"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent"></div>
                  <div className="absolute top-3 left-3 flex gap-2">
                    <span className="bg-slate-900/80 backdrop-blur-sm px-2.5 py-1 rounded-md text-white text-xs font-medium">
                      {project.category}
                    </span>
                    <span className="bg-white/90 px-2.5 py-1 rounded-md text-gray-900 text-xs font-medium">
                      {project.year}
                    </span>
                  </div>
                </div>
                <div className="p-5">
                  <h3 className="text-lg font-semibold text-gray-900 mb-2 group-hover:text-slate-700 transition">
                    {project.title}
                  </h3>
                  <p className="text-gray-600 text-sm mb-3 line-clamp-2">{project.shortDesc}</p>
                  <div className="flex flex-wrap gap-1.5 mb-3">
                    {project.tags.slice(0, 3).map((tag, idx) => (
                      <span key={idx} className="bg-slate-100 px-2 py-1 rounded text-xs text-gray-700">
                        {tag}
                      </span>
                    ))}
                    {project.tags.length > 3 && (
                      <span className="bg-slate-100 px-2 py-1 rounded text-xs text-gray-700">
                        +{project.tags.length - 3}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center text-slate-700 text-sm font-medium">
                    Read more
                    <ChevronRight size={16} className="ml-1 group-hover:translate-x-1 transition-transform" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Skills Section */}
      <div id="skills" className="py-20">
        <div className="max-w-6xl mx-auto px-6 lg:px-8">
          <div className="mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-3">Technical Skills</h2>
            <p className="text-gray-600 max-w-2xl">
              A comprehensive toolkit for building end-to-end AI/ML solutions
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {skills.map((skillGroup, idx) => (
              <div key={idx} className="bg-white border border-gray-200 p-6 rounded-lg">
                <h3 className="font-semibold text-gray-900 mb-3">{skillGroup.category}</h3>
                <div className="flex flex-wrap gap-2">
                  {skillGroup.items.map((skill, i) => (
                    <span key={i} className="bg-slate-100 text-slate-700 px-3 py-1 rounded-md text-sm">
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

{/* About Section */}
<div id="about" className="bg-slate-50 py-20">
  <div className="max-w-6xl mx-auto px-6 lg:px-8">
    <div className="max-w-3xl">
      <h2 className="text-3xl font-bold text-gray-900 mb-6">
        Why SIT Applied AI?
      </h2>
      <div className="space-y-4 text-gray-700 leading-relaxed">
        <p>
          My journey to Applied AI has been shaped by diverse experiences that taught me discipline, 
          innovation, and the importance of real world impact. As a member of the Temasek Polytechnic 
          Tennis Team (2024/25 2025/26), I helped my team earn a Silver medal at the Tennis POL ITE 
          tournament. More notably, in 2025/26, the TP men's tennis team achieved their first gold medal 
          in 15 years a historic milestone that demonstrated what dedication and teamwork can accomplish. 
          These experiences on the court taught me discipline, teamwork, and perseverance, qualities that 
          translate directly to solving complex technical challenges in AI development.
        </p>
        <p>
          SIT's Applied AI program stands out to me because it emphasizes practical, industry integrated 
          learning. The curriculum's focus on real world projects, industry partnerships, and applied research 
          aligns perfectly with my hands on approach to learning. I'm drawn to SIT's Integrated Work Study 
          Programme, which will allow me to apply AI concepts in actual business environments and contribute 
          to real solutions.
        </p>
        <p>
          My portfolio demonstrates that I don't just build projects for academic credit I create systems 
          designed for real deployment, validated by actual stakeholders like the SAFRA Tampines Club Manager. 
          I want to continue this trajectory at SIT, collaborating with faculty and industry partners to develop 
          AI solutions that address genuine business challenges. Whether it's improving healthcare diagnostics, 
          enhancing manufacturing efficiency, or creating more accessible technology, I'm committed to using AI 
          as a force for positive change.
        </p>
        <p>
          SIT's Applied AI program represents the next step in my journey a place where my technical skills, 
          competitive drive, and commitment to meaningful innovation can converge to create real impact.
        </p>
      </div>
    </div>
  </div>
</div>


      {/* Contact Section */}
      <div id="contact" className="py-20">
        <div className="max-w-6xl mx-auto px-6 lg:px-8">
          <div className="bg-slate-900 rounded-2xl p-12 text-center">
            <h2 className="text-3xl font-bold text-white mb-4">Let's Connect</h2>
            <p className="text-slate-300 mb-8 max-w-2xl mx-auto">
              Interested in discussing AI projects or potential collaborations? 
              I'd love to hear from you.
            </p>
            <div className="flex flex-wrap gap-3 justify-center">
              <a
                href="mailto:megacertgt@gmail.com"
                className="flex items-center gap-2 bg-white text-slate-900 px-6 py-3 rounded-lg font-medium hover:bg-slate-100 transition"
              >
                <Mail size={18} />
                Email Me
              </a>
              <a
                href="https://www.linkedin.com/in/joel-tan1245"
                className="flex items-center gap-2 bg-slate-800 text-white px-6 py-3 rounded-lg font-medium hover:bg-slate-700 transition"
              >
                <Linkedin size={18} />
                LinkedIn
              </a>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t border-gray-200 py-8">
        <div className="max-w-6xl mx-auto px-6 lg:px-8 text-center">
          <p className="text-sm text-gray-500">© 2025 Joel Tan. Built with React.</p>
        </div>
      </footer>
    </div>
  );
}