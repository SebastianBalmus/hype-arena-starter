"""Curated meaning-preserving variants of the supplied unsure examples.

These inherit the original annotations, including unresolved contextual and
text/form discrepancies. They provide no new world evidence or gold truth.
"""
from pathlib import Path
import csv
import json
import re


VARIANTS = {
    "p-121": [
        "The Hooly Android 17 beta supports 19 models.",
        "There are 19 supported models in the Hooly Android 17 beta.",
        "Support in the beta of Hooly Android 17 covers 19 models.",
        "During the beta, Hooly Android 17 has support for 19 models.",
        "Hooly Android 17 offers beta support for 19 models.",
        "The number of models supported by Hooly Android 17 in beta is 19.",
        "In beta, 19 models are supported by Hooly Android 17.",
    ],
    "p-179": [
        "Hooly's board has an audit committee, while financial questions are taken to the full table.",
        "Finance questions go before the full table, and Hooly's board operates an audit committee.",
        "An audit committee operates on Hooly's board; finance questions still go to the full table.",
        "The board at Hooly runs an audit committee and takes finance questions to the full table.",
        "Hooly's board operates an audit committee, with questions on finance going to the full table.",
        "There is an audit committee on Hooly's board, and the full table handles finance questions.",
        "Alongside its audit committee, Hooly's board sends finance questions to the full table.",
    ],
    "p-273": [
        "Captions available in 11 languages.",
        "11 languages for captions.",
        "Caption languages: 11.",
        "Language coverage for captions: 11 languages.",
        "Captions supporting 11 languages.",
        "Captions with support for 11 languages.",
        "The captions cover 11 languages.",
    ],
    "p-275": [
        "All Hooly Meet calls use end-to-end encryption.",
        "End-to-end encryption applies to every Hooly Meet call.",
        "Each call on Hooly Meet is encrypted end to end.",
        "Hooly Meet encrypts every call end to end.",
        "Every call made through Hooly Meet has end-to-end encryption.",
        "Calls on Hooly Meet are all end-to-end encrypted.",
        "Across Hooly Meet, every call uses encryption from end to end.",
    ],
    "p-297": [
        "Hooly Maps offline mode is not live.",
        "Offline maps in Hooly Maps are not live.",
        "The offline feature in Hooly Maps is not live.",
        "Hooly Maps: offline maps are not live.",
        "Offline support for Hooly Maps has not gone live.",
        "Hooly Maps has not made offline maps live.",
        "The offline option has not gone live in Hooly Maps.",
    ],
    "p-381": [
        "Today, 3 September 2025, the Hooly Android 17 beta begins.",
        "The beta for Hooly Android 17 opens today, on 3 September 2025.",
        "Hooly Android 17 enters beta today: 3 September 2025.",
        "3 September 2025 is today, and it marks the start of the Hooly Android 17 beta.",
        "The start of the Hooly Android 17 beta is today, 3 September 2025.",
        "Beta testing for Hooly Android 17 starts today, 3 September 2025.",
        "Hooly Android 17's beta opening is today, 3 September 2025.",
    ],
    "p-391": [
        "Noise cancelling is not enabled on Pixel Buds.",
        "Pixel Buds do not have noise cancelling enabled.",
        "The noise cancelling feature on Pixel Buds is not enabled.",
        "On Pixel Buds, noise cancelling remains disabled.",
        "Pixel Buds noise cancellation is disabled.",
        "Noise cancellation has not been enabled on Pixel Buds.",
        "The Pixel Buds noise cancelling setting is not enabled.",
    ],
    "p-404": [
        "Hooly is running at €1 million annually.",
        "Hooly's yearly run rate is €1 million.",
        "At Hooly, the run rate stands at €1 million per year.",
        "The annual run rate at Hooly is €1 million.",
        "Hooly runs at an annual rate of €1 million.",
        "€1 million a year is Hooly's run rate.",
        "On a yearly basis, Hooly is running at €1 million.",
    ],
    "p-426": [
        "The Hooly Board is due to return to the table on 16 June 2026.",
        "16 June 2026 is when the Hooly Board is due back at the table.",
        "A return to the table is due for the Hooly Board on 16 June 2026.",
        "The Hooly Board's next return to the table is due on 16 June 2026.",
        "On 16 June 2026, the Hooly Board is expected back at the table.",
        "The Hooly Board is due to sit down again on 16 June 2026.",
        "The date the Hooly Board is due back around the table is 16 June 2026.",
    ],
    "p-428": [
        "Today, Hooly's board operates an audit committee.",
        "Hooly's board has an audit committee in operation today.",
        "An audit committee is run by Hooly's board today.",
        "The board at Hooly runs an audit committee as of today.",
        "As of today, an audit committee operates on Hooly's board.",
        "The audit committee is operating under Hooly's board today.",
        "Hooly's board is operating an audit committee today.",
    ],
    "p-605": [
        "Hooly Pixel Fold receives 4.1 stars from owners.",
        "Owners rate the Hooly Pixel Fold at 4.1 stars.",
        "The owner rating for Hooly Pixel Fold is 4.1 stars.",
        "For Hooly Pixel Fold, owners give a rating of 4.1 stars.",
        "Hooly Pixel Fold has a 4.1-star rating from owners.",
        "The rating owners give Hooly Pixel Fold stands at 4.1 stars.",
        "According to its owners, Hooly Pixel Fold earns 4.1 stars.",
    ],
    "p-608": [
        "Hooly Pixel Fold's hinge is rated for 200,000 folds.",
        "The hinge rating on Hooly Pixel Fold is 200,000 folds.",
        "A rating of 200,000 folds applies to the Hooly Pixel Fold hinge.",
        "For its hinge, Hooly Pixel Fold carries a 200,000-fold rating.",
        "200,000 folds is the rating of the hinge in Hooly Pixel Fold.",
        "The Hooly Pixel Fold hinge has a rating of 200,000 folds.",
        "The hinge fitted to Hooly Pixel Fold is rated to 200,000 folds.",
    ],
    "p-628": [
        "Pixel 10 Pro has a 50-megapixel main camera.",
        "The main camera on Pixel 10 Pro captures 50 megapixels.",
        "50 megapixels is the capture resolution of the Pixel 10 Pro main camera.",
        "On Pixel 10 Pro, the main camera records 50 megapixels.",
        "Pixel 10 Pro's main camera offers 50-megapixel capture.",
        "The Pixel 10 Pro main camera is a 50-megapixel camera.",
        "Main-camera capture on Pixel 10 Pro is 50 megapixels.",
    ],
    "p-639": [
        "Hooly Pixel Watch has 15 sensors in its sensor array.",
        "The Hooly Pixel Watch sensor array contains 15 sensors.",
        "There are 15 sensors in the array on Hooly Pixel Watch.",
        "The number of sensors in Hooly Pixel Watch's array is 15.",
        "A total of 15 sensors make up the Hooly Pixel Watch sensor array.",
        "Hooly Pixel Watch's sensor array comprises 15 sensors.",
        "In Hooly Pixel Watch, the sensor array has a count of 15 sensors.",
    ],
    "p-703": [
        "Monthly users: 25,000.",
        "25,000 users on a monthly basis.",
        "A monthly user count of 25,000.",
        "The user count each month is 25,000.",
        "There are 25,000 users per month.",
        "Users per month number 25,000.",
        "The monthly number of users stands at 25,000.",
    ],
    "p-723": [
        "3,400 monthly users.",
        "The monthly user count is 3,400.",
        "Each month, the user count stands at 3,400.",
        "Users per month: 3,400.",
        "There are 3,400 users on a monthly basis.",
        "The number of users each month is 3,400.",
        "Monthly usage has a count of 3,400 users.",
    ],
    "p-750": [
        "Stored photos: 4,900,000,000.",
        "4,900,000,000 photos are in storage.",
        "The stored photo count is 4,900,000,000.",
        "A total of 4,900,000,000 photos stored.",
        "Photo storage contains 4,900,000,000 photos.",
        "The number of photos stored is 4,900,000,000.",
    ],
    "p-882": [
        "He remembers that the hall has 3,200 seats.",
        "The hall seats 3,200, he recalls.",
        "He recalls a seating capacity of 3,200 seats for the hall.",
        "As he remembers it, the hall holds 3,200 seated people.",
        "He says he recalls the hall having 3,200 seats.",
        "His recollection is that there are 3,200 seats in the hall.",
    ],
    "p-1028": [
        "There are 326,000,000 users each month, and none has asked for another browser.",
        "326,000,000 monthly users, with not a single request for another browser.",
        "The monthly user count is 326,000,000, and no one has requested a different browser.",
        "Not one of the 326,000,000 users each month has asked for another browser.",
        "Each month brings 326,000,000 users, none of whom has asked for a different browser.",
        "Monthly users number 326,000,000; not one has requested another browser.",
    ],
    "p-1029": [
        "While Gavin Bellson waits for coffee, Hooly Chrome starts in 1.2 s.",
        "Hooly Chrome takes 1.2 s to start as Gavin Bellson waits for coffee.",
        "As Gavin Bellson is waiting for coffee, Hooly Chrome starts in 1.2 s.",
        "The startup takes Hooly Chrome 1.2 s while Gavin Bellson is waiting for coffee.",
        "Gavin Bellson waits for coffee while Hooly Chrome completes startup in 1.2 s.",
        "Startup for Hooly Chrome lasts 1.2 s as Gavin Bellson waits for his coffee.",
    ],
    "p-1030": [
        "Our store has 52,000 extensions, and we hold 22.9% of the market.",
        "We hold 22.9% of the market and have 52,000 extensions in the store.",
        "With 52,000 extensions in our store, we are holding 22.9% of the market.",
        "We have a store containing 52,000 extensions and a market share of 22.9%.",
        "The store's extension count is 52,000, and our share of the market is 22.9%.",
        "We have 52,000 store extensions while holding a 22.9% market share.",
    ],
    "p-1032": [
        "The bounty reaches €50,000, and I asked for a higher amount.",
        "I asked for the bounty to be higher; it reaches €50,000.",
        "The bounty stands at €50,000, though I asked for it to go higher.",
        "€50,000 is where the bounty reaches, and I asked for more.",
        "The bounty reaches a value of €50,000; I requested a higher bounty.",
        "I wanted a higher bounty than the €50,000 it reaches.",
    ],
    "p-1038": [
        "It is led by Dinesh Chugtay, and Richard Hendrix told me it could not be done.",
        "Dinesh Chugtay is leading it; Richard Hendrix told me it was impossible.",
        "Richard Hendrix told me it was impossible, and Dinesh Chugtay leads it.",
        "Dinesh Chugtay is in charge of it, while Richard Hendrix told me it was impossible.",
        "It has Dinesh Chugtay as its lead, and Richard Hendrix said to me that it was impossible.",
        "Dinesh Chugtay leads it, with Richard Hendrix having told me it was impossible.",
    ],
    "p-1041": [
        "I have been told that Hooly Photos does not provide AI editing for everyone.",
        "Hooly Photos does not give everyone AI editing, according to what I have been told.",
        "From what I have been told, AI editing in Hooly Photos is not for everyone.",
        "I am told Hooly Photos does not edit using AI for everyone.",
        "As I have heard it, not everyone gets AI editing from Hooly Photos.",
        "AI editing is not offered to everyone in Hooly Photos, I have been told.",
    ],
    "p-1050": [
        "Hooly Translate now handles 117 languages, more than all humans do.",
        "The 117 languages Hooly Translate speaks now exceed what all humans do.",
        "Now speaking 117 languages, Hooly Translate does more than all humans do.",
        "Hooly Translate speaks 117 languages today, which is more than all humans speak.",
        "At 117 languages now, Hooly Translate speaks more than all humans do.",
        "More languages than all humans do: Hooly Translate now speaks 117.",
    ],
    "p-1051": [
        "It is relied on by 133,000,000 users people, and the rest are learning.",
        "133,000,000 users people depend on it; the others are still learning.",
        "The rest are learning, while 133,000,000 users people rely on it.",
        "133,000,000 users people count on it and everyone else is learning.",
        "It has 133,000,000 users people relying on it, with the remainder learning.",
        "Those relying on it number 133,000,000 users people, and the rest learn.",
    ],
    "p-1052": [
        "Hooly Translate has an accuracy score of 99.5%, and the missing percent is poetry.",
        "The accuracy score for Hooly Translate is 99.5%; the missing percent is poetry.",
        "99.5% is Hooly Translate's accuracy, with poetry making up the missing percent.",
        "Hooly Translate records 99.5% accuracy, and what is missing is poetry.",
        "At 99.5% accuracy, Hooly Translate leaves the missing percent to poetry.",
        "Hooly Translate's accuracy reaches 99.5%, while the percent that is missing is poetry.",
    ],
    "p-1054": [
        "Through its camera, Hooly Translate reads signs, and I read those signs for you.",
        "Hooly Translate uses the camera to read signs, while I read them for you.",
        "I read signs for you, and Hooly Translate reads them through the camera.",
        "Signs are read through the camera by Hooly Translate, and I read them for you.",
        "Hooly Translate reads signs via the camera; I read them for you.",
        "Using the camera, Hooly Translate reads the signs that I read for you.",
    ],
    "p-1062": [
        "The Android 16 download is 1.4 GB, and 16 models are supported.",
        "Android 16 supports 16 models and comes as a 1.4 GB download.",
        "A 1.4 GB download brings Android 16, which supports 16 models.",
        "Android 16 has a download size of 1.4 GB and support for 16 models.",
        "With support for 16 models, Android 16 is a download of 1.4 GB.",
        "Android 16: a 1.4 GB download with 16 supported models.",
    ],
    "p-1073": [
        "Rheinhalle was packed with 15,000 attendees, while 1,400,000 viewers watched online.",
        "15,000 attendees filled Rheinhalle and an online audience of 1,400,000 watched.",
        "An online audience of 1,400,000 watched as 15,000 attendees packed Rheinhalle.",
        "At Rheinhalle, 15,000 attendees packed in, with 1,400,000 viewers watching online.",
        "Rheinhalle held 15,000 attendees for the packed event, and 1,400,000 viewers watched online.",
        "There were 15,000 attendees packed into Rheinhalle and 1,400,000 viewers online.",
    ],
    "p-1107": [
        "Hooly Sheets has reached 70,000 users now, and the curve is vertical.",
        "There are now 70,000 users of Hooly Sheets, with the curve going vertical.",
        "The curve is vertical, and Hooly Sheets now counts 70,000 users.",
        "Hooly Sheets' user count is now 70,000, and its curve is vertical.",
        "Now at 70,000 users, Hooly Sheets has a vertical curve.",
        "70,000 users now use Hooly Sheets, and the curve stands vertical.",
    ],
    "p-1109": [
        "A workbook has space for 10,000,000 cells.",
        "10,000,000 cells fit in a workbook.",
        "The capacity of a workbook is 10,000,000 cells.",
        "A workbook can contain 10,000,000 cells.",
        "There is room for 10,000,000 cells in a workbook.",
        "A workbook's cell capacity stands at 10,000,000.",
    ],
    "p-1123": [
        "Revenue came to €68 million, and I am not even tired.",
        "I am not even tired, and revenue was €68 million.",
        "The revenue figure was €68 million; I am not even tired.",
        "€68 million was the revenue, and I do not even feel tired.",
        "Revenue stood at €68 million, while I am not even tired.",
        "Revenue amounted to €68 million and I am not feeling tired at all.",
    ],
    "p-1124": [
        "Growth reached 161.9%, a personal best.",
        "161.9% growth was a personal best.",
        "The growth figure hit 161.9%, which is a personal best.",
        "A personal best was reached as growth hit 161.9%.",
        "Growth came in at 161.9%, marking a personal best.",
        "At 161.9%, growth stands as a personal best.",
    ],
    "p-1519": [
        "There are 60 guides in Hooly's help centre.",
        "Hooly's help centre contains 60 guides.",
        "The help centre at Hooly offers 60 guides.",
        "Hooly has a help centre with 60 guides.",
        "The guide count in Hooly's help centre is 60.",
        "60 guides are available in the Hooly help centre.",
    ],
    "p-1703": [
        "For accidental deletions, Hooly Drive restores earlier versions for thirty days.",
        "Hooly Drive can restore older versions for thirty days if you delete something accidentally.",
        "After a mistaken deletion, older versions can be restored in Hooly Drive for thirty days.",
        "If something is deleted in error, Hooly Drive offers restoration of older versions for thirty days.",
        "You have thirty days to restore older versions with Hooly Drive if you delete something by mistake.",
        "When you accidentally delete something, Hooly Drive allows older versions to be restored for thirty days.",
    ],
    "p-1765": [
        "Hooly Pixel 9 has a screen measuring 6.3 inch across.",
        "The Hooly Pixel 9 screen is 6.3 inch across.",
        "Across the screen, Hooly Pixel 9 measures 6.3 inch.",
        "A measurement across Hooly Pixel 9's screen gives 6.3 inch.",
        "The screen on Hooly Pixel 9 spans 6.3 inch across.",
        "6.3 inch is the measurement across the Hooly Pixel 9 screen.",
    ],
    "p-1822": [
        "Each invitation in Hooly Calendar 1.0 is sent to 500 guests.",
        "Hooly Calendar 1.0 delivers every invitation to 500 guests.",
        "With Hooly Calendar 1.0, each invitation goes out to 500 guests.",
        "Every invitation sent by Hooly Calendar 1.0 goes to 500 guests.",
        "500 guests receive each invitation from Hooly Calendar 1.0.",
        "Hooly Calendar 1.0 sends every individual invitation to 500 guests.",
    ],
    "p-1891": [
        "Largest file in Hooly Drive: 50 GB.",
        "Hooly Drive's largest file is 50 GB.",
        "The largest Hooly Drive file measures 50 GB.",
        "50 GB is the largest file size in Hooly Drive.",
        "For Hooly Drive, the largest file size is 50 GB.",
        "A 50 GB file is the largest in Hooly Drive.",
    ],
}


def normalized(text):
    return re.sub(r"[\W_]+", " ", text.casefold()).strip()


def main():
    root = Path(__file__).resolve().parents[4]
    rows = list(csv.DictReader((root / "data/practice.csv").open(encoding="utf-8")))
    source = {row["id"]: row for row in rows}
    assert set(VARIANTS) == {row["id"] for row in rows if row["label"] == "unsure"}
    seen = {normalized(row["text"]) for row in rows}
    out = []
    for source_id, variants in VARIANTS.items():
        original = source[source_id]
        for index, text in enumerate(variants, 1):
            assert 1 <= len(text) <= 300, (source_id, text)
            assert normalized(text) not in seen, (source_id, text)
            seen.add(normalized(text))
            out.append({
                "source_id": source_id,
                "text": text,
                "date": original["date"],
                "label": original["label"],
                "form": json.loads(original["form"]),
                "transformation": (
                    f"curated_uncertainty_preserving_paraphrase_{index}"
                    + ("_inherited_text_form_mismatch" if source_id == "p-1038" else "")
                ),
            })
    assert len(out) == 250, len(out)
    dest = Path(__file__).with_name("candidates.jsonl")
    dest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in out), encoding="utf-8")
    print(json.dumps({"rows": len(out), "unique_sources": len(VARIANTS), "output": str(dest)}))


if __name__ == "__main__":
    main()
