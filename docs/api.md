# API reference

Interactive OpenAPI is served at `/docs`.

| Method | Endpoint                                 | Purpose                                 |
| ------ | ---------------------------------------- | --------------------------------------- |
| GET    | `/api/health`                            | API and configured model readiness      |
| POST   | `/api/videos`                            | Multipart video upload (`file`, `role`) |
| GET    | `/api/videos`                            | List uploaded video metadata            |
| POST   | `/api/jobs`                              | Create and enqueue a processing job     |
| GET    | `/api/jobs`                              | List recent jobs                        |
| GET    | `/api/jobs/{id}`                         | Job status, progress and diagnostics    |
| GET    | `/api/jobs/{id}/events`                  | Event records and evidence URLs         |
| GET    | `/api/jobs/{id}/export.json`             | Full event and diagnostics JSON         |
| GET    | `/api/jobs/{id}/export.csv`              | Flat event export                       |
| GET    | `/evidence/{job_id}/evidence/{filename}` | Stored evidence file                    |

## Create job example

```json
{
	"traffic_video_id": "<uploaded-video-id>",
	"rear_video_id": null,
	"config": {
		"mode": "same_camera",
		"stop_line": [
			{ "x": 0.08, "y": 0.52 },
			{ "x": 0.92, "y": 0.52 }
		],
		"approach_side": "either",
		"direction": "northbound",
		"signal_timeline": [
			{
				"start_seconds": 0,
				"end_seconds": 10,
				"color": "green",
				"direction": "northbound"
			},
			{
				"start_seconds": 10,
				"end_seconds": 20,
				"color": "red",
				"direction": "northbound"
			}
		],
		"frame_stride": 1,
		"min_track_observations": 3
	}
}
```

The sample timestamps and coordinates are illustrative. Do not use them as ground truth for a real video.

For `separate_cameras`, supply `rear_video_id`, `rear_time_offset_seconds`, and `min_travel_seconds` / `max_travel_seconds`. A separate-camera event is not associated unless the candidate passes both temporal and visual similarity tests.

## Export fields

Each JSON event includes event ID, crossing timestamp, traffic/rear track IDs, direction, signal color/status/reason, classification/reason, association status/confidence/reason, plate text/status/confidence/observation count/conflict count, traffic frame URL and plate evidence URLs. CSV exposes the principal event, signal, association, plate and evidence fields.

## Error handling

- `400`: uploaded bytes are not a readable video
- `413`: upload exceeds `MOTOSIGNAL_MAX_UPLOAD_MB`
- `415`: extension is not allowed
- `422`: invalid camera role, config or job request
- `404`: unknown video or job identifier

The API currently has no authentication.
