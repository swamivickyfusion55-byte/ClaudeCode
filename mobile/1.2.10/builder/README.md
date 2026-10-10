---
title: APKBuilder
emoji: 🤖
colorFrom: blue
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Phoenix APK Builder 1.2.10

Upload every file in this pack, including `source.zip`.
Then open the Space and press Build APK.

**1.2.10** fixes "Push to all Spaces" (`Updated 0, failed 21 … HTTP 400 …`): the commit request now goes out in
exactly the form Hugging Face's own clients send. Same files as 1.2.9b — upload all of them, including
`source.zip`, replacing the old ones.
