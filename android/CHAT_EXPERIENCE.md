# Hu Tao's single-thread conversation experience

## Product requirement

The companion UI is one persistent, WhatsApp-style conversation with Hu Tao. Do not create a list of separate chats, temporary sessions, or conversations that reset when the app closes.

## Conversation behaviour

- Every user message and every Hu Tao reply is appended to one local chronological transcript.
- The transcript is saved to the device immediately after each turn.
- Reopening the app restores the same conversation at the latest message.
- The user can scroll to older messages and search the entire history.
- No automatic chat summarization is allowed to replace or delete original messages.
- Optional derived memories are stored separately; they supplement the transcript rather than replacing it.
- When responding, retrieve recent messages and older relevant messages. Do not stuff the entire transcript into the active model prompt.
- If a message cannot be stored, clearly show a local save error rather than pretending it was remembered.

## Memory layers

1. **Complete transcript:** immutable by default, searchable; all user and assistant messages.
2. **Personal facts and preferences:** durable structured records extracted from conversation, with correction and deletion.
3. **World knowledge taught by the user:** separate facts about the user's world, each attributed to the user and searchable.
4. **Teyvat autobiographical/lore memories:** the staged source-backed Hu Tao life and news reports.

The user should be able to inspect and correct derived memories. The raw transcript must remain available until explicitly deleted.

## Offline and storage

Conversation, learned facts, life memories and the model run locally. No cloud conversation service is required. Storage is finite, so show database size and provide backup/export and explicit delete controls. Later releases should encrypt private databases at rest and let the user choose a backup destination.

## Context assembly

The model receives the latest few messages, older matching transcript passages, relevant personal memory, relevant user-taught world facts, and relevant Hu Tao life memories/news. The complete local history is not the same as the active model context window.
