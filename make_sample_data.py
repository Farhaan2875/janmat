"""Builds data/sample_econsultation_comments.csv

IMPORTANT: comments submitted on the MCA e-consultation portal are not
published as a downloadable dataset, so this file is a SIMULATED demo set
written for a fictional "Draft Digital Data Governance Rules". Each comment
is stitched from hand-written sentences about five themes, with a known
sentiment, so we can measure how well models trained on tweets transfer to
policy language. To use real comments, upload your own CSV in the dashboard.
"""
import random

import pandas as pd

random.seed(7)

# theme -> sentiment -> hand-written sentences
SENTENCES = {
    "Privacy and consent": {
        "positive": [
            "The consent requirements finally give ordinary citizens real control over their personal data.",
            "I welcome the clear rule that companies must ask permission before collecting my information.",
            "Plain-language consent notices are a big step forward for digital rights in India.",
            "The right to withdraw consent at any time is a strong and much needed safeguard.",
            "This draft protects the privacy of common people, especially those new to the internet.",
        ],
        "negative": [
            "The consent clause is vague and lets companies bury permissions inside long notices nobody reads.",
            "Broad exemptions for the government undermine the privacy protections promised to citizens.",
            "I am worried that consent will become a meaningless tick-box exercise under these rules.",
            "There is no clear limit on how long companies may store personal data, which is unacceptable.",
            "The draft fails to protect the data of children and elderly users who cannot judge consent notices.",
        ],
        "neutral": [
            "The draft defines consent as a free, specific and informed agreement by the individual.",
            "Section 4 lists the purposes for which personal data may be processed.",
            "Please clarify whether consent managers will be registered with a central authority.",
            "The rules mention notice requirements, but the format of the notice is not given.",
            "Could the ministry explain how consent will apply to data collected before the rules come into force?",
        ],
    },
    "Burden on startups and MSMEs": {
        "positive": [
            "The phased compliance timeline is practical and gives small businesses time to prepare.",
            "I appreciate the lighter obligations for startups, which will help local innovation grow.",
            "Clear compliance steps will reduce confusion and make it easier for MSMEs to build customer trust.",
            "The exemption for very small data fiduciaries is a sensible and balanced decision.",
            "Standard templates for privacy policies will save small shops a lot of legal cost.",
        ],
        "negative": [
            "These compliance costs will crush small startups that cannot afford a dedicated legal team.",
            "The reporting duties are too heavy for MSMEs and will push many founders out of business.",
            "Small businesses were not consulted properly, and the rules ignore how we actually operate.",
            "Mandatory audits are expensive and unfair to companies with only a handful of employees.",
            "The draft favours large technology firms and leaves small enterprises struggling to comply.",
        ],
        "neutral": [
            "The draft says significant data fiduciaries will have additional obligations.",
            "Please specify the threshold that separates a small data fiduciary from a significant one.",
            "The rules require appointing a point of contact for data protection queries.",
            "It is not stated whether sole proprietorships fall under the same compliance schedule.",
            "Could the ministry share the estimated compliance cost for a business with ten employees?",
        ],
    },
    "Penalties and enforcement": {
        "positive": [
            "Strong financial penalties will finally force companies to take data breaches seriously.",
            "I support the graded penalty structure because it matches the penalty to the seriousness of the breach.",
            "Mandatory breach notification to affected users is a transparent and responsible requirement.",
            "Holding companies accountable for leaks is long overdue and the draft does it well.",
            "The enforcement framework is firm but fair and should deter careless data handling.",
        ],
        "negative": [
            "The penalty amounts are arbitrary and so high that they will discourage investment in India.",
            "There is no clarity on how penalties will be calculated, which invites misuse and harassment.",
            "Enforcement appears weak, because breaches are only punished after long and slow inquiries.",
            "Government bodies are exempt from penalties, which is deeply unfair to private companies.",
            "The draft punishes companies heavily but offers nothing to compensate the citizens who suffer harm.",
        ],
        "neutral": [
            "The schedule lists maximum penalties for different categories of violation.",
            "Penalties are imposed after an inquiry by the board according to the draft.",
            "Please clarify who decides the amount of a penalty in case of repeated breaches.",
            "The rules refer to voluntary undertakings as an alternative to penalties.",
            "It would help to know whether penalties apply per incident or per affected user.",
        ],
    },
    "Grievance redressal and oversight": {
        "positive": [
            "A dedicated board to hear complaints will give citizens a real place to seek justice.",
            "The digital-first complaint process is convenient and saves people a trip to a court.",
            "I am glad that companies must respond to grievances within a fixed time period.",
            "Independent oversight of data protection is a positive and necessary reform.",
            "The appeal option against board decisions makes the grievance process feel fair and transparent.",
        ],
        "negative": [
            "The board is appointed by the government, so it cannot be seen as truly independent.",
            "Citizens have to approach the company first, and in practice nobody ever answers those complaints.",
            "The grievance process is too slow and gives rural citizens without internet access no real remedy.",
            "There is no provision for compensation, so complaining to the board achieves very little.",
            "The oversight body lacks technical staff and funding, so it will fail to handle thousands of cases.",
        ],
        "neutral": [
            "The draft describes the composition and term of the data protection board.",
            "Complaints must be filed through a digital portal as per the proposed rules.",
            "Please clarify whether the board will have regional offices.",
            "The rules state the board may refer a dispute to mediation.",
            "Could the ministry explain the qualifications required for board members?",
        ],
    },
    "Awareness and implementation": {
        "positive": [
            "The planned public awareness campaign will help citizens understand their new rights.",
            "Rolling the rules out in stages is a thoughtful approach that reduces disruption.",
            "I like that the draft promises guidance material in multiple Indian languages.",
            "Training programmes for officials will make the implementation smoother and more consistent.",
            "The draft shows real intent to involve states and local bodies in implementation.",
        ],
        "negative": [
            "Most citizens in villages have never heard of these rules, and no awareness plan is described.",
            "The implementation timeline is unrealistic and will cause chaos for businesses and officials alike.",
            "Rules published only in English exclude millions of people who need protection the most.",
            "There is no budget allocated for implementation, so the whole scheme looks like an empty promise.",
            "Officials in district offices are untrained, and nothing here explains how they will learn the rules.",
        ],
        "neutral": [
            "The draft proposes that the rules come into force in phases after final notification.",
            "Please share the planned date for the final notification of the rules.",
            "The ministry may issue guidance documents from time to time according to the draft.",
            "It is not clear which department will coordinate implementation across states.",
            "Could the ministry publish a roadmap of implementation milestones?",
        ],
    },
}

OPENERS = ["", "", "", "As a concerned citizen, ", "Speaking for our association, ", "From a legal viewpoint, ",
           "In my experience, ", "On behalf of our organisation, "]
COMMENTERS = ["Citizen", "Citizen", "Citizen", "Industry body", "Academic", "Legal professional", "NGO"]

rows = []
seen = set()
# weights: more negative/neutral than positive, closer to real consultations
sentiment_pool = ["negative"] * 5 + ["neutral"] * 3 + ["positive"] * 3
while len(rows) < 360:
    topic = random.choice(list(SENTENCES))
    sentiment = random.choice(sentiment_pool)
    pool = SENTENCES[topic][sentiment]
    parts = random.sample(pool, random.choice([1, 2]))
    opener = random.choice(OPENERS)
    body = " ".join(parts)
    if opener:
        if not body.startswith("I "):
            body = body[0].lower() + body[1:]   # "In my experience, the board ..."
    text = opener + body
    if text in seen:
        continue
    seen.add(text)
    rows.append({"comment_id": len(rows) + 1, "commenter_type": random.choice(COMMENTERS),
                 "comment": text, "true_label": sentiment, "true_topic": topic})

df = pd.DataFrame(rows)
df.to_csv("data/sample_econsultation_comments.csv", index=False)
print(df["true_label"].value_counts().to_string())
print("Saved", len(df), "comments")
