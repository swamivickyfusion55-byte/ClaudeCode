# Dual Hugging Face Space support

The Android client now supports two independently configured Phoenix Hugging Face Spaces.

## How to use

1. Open **Connection & API settings**.
2. Select **Space 1** and enter its name and HF Space URL. Space 1 defaults to the existing SwamitechUAT URL.
3. Select **Space 2** and enter its name and HF Space URL.
4. Keep the same submit endpoint if both Spaces expose the same Phoenix API contract.
5. Start the first video job with Space 1 selected.
6. Switch the **Processing Space** dropdown to Space 2 and start another job.
7. Open **History**. Each job has its own Space name, remote job ID, status, progress and ETA.

## Parallelism

A maximum of two cloud jobs are monitored concurrently. Each job captures its selected Space configuration at start time, so changing the dropdown after starting a job does not redirect that job to another Space.

History entries are persisted locally. If the app is restarted while a remote job is still in a tracked state, the app attempts to resume status polling using the saved Space URL and remote job ID.

Stopping monitoring only stops the Android client's polling. It does not claim to cancel a remote Phoenix job because the supplied API contract does not expose a verified cancellation endpoint.
