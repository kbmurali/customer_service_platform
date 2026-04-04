#%%
import requests

BASE_URL = "http://localhost:3001"
AUTH = ("pk-lf-a4bf61a1-b043-446c-8350-2f14ba66f110", "sk-lf-69f4e346-705f-4cf2-a66c-095636bfcdfc")
SESSION_COOKIE = "eeyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..fjhLQ-dJgKdEHwWD.O-jX0GA0-YKnJ9vj55W6DZPNqQucM1U_wxCNcdrdxr8lTd_GgdReX__lEGoAeDrIfHc2crJMnj2rjpyMLkl_cSnDq2negkBao3B5BU2kMHhOzdNGKWgFqkNvXMD5FKuL_CVmU_6TQhza34CcUodkNFT85JWTxklkXmdvcnEKOXDA80BssLEYHvw1XhnbkucWAbeV9JbMuBfG593nraZTK1OH0hGo6Xq9DTObfKfm.LrJEGDBkl7PC4Bkb19G6SQ"
PROJECT_ID = "cmlyeepuc0006mswyg179kqye"

page = 1
count = 0

#%%
while True:
    res = requests.get(f"{BASE_URL}/api/public/traces", auth=AUTH, params={
        "toTimestamp": "2026-03-14T00:00:00Z",
        "limit": 100,
        "page": page
    })
    
    traces = res.json().get("data", [])
   
    if not traces:
        break

    trace_ids = [trace['id'] for trace in traces]
    
    # tRPC batch format: body wrapped in {"0": {"json": {...}}}
    delete_res = requests.post(
        f"{BASE_URL}/api/trpc/traces.deleteMany?batch=1",
        cookies={"next-auth.session-token": SESSION_COOKIE},
        json={
            "0": {
                "json": {
                    "projectId": PROJECT_ID,
                    "traceIds": trace_ids
                }
            }
        }
    )
    count += len(traces)
    print(f">>>>> Deleted {count} traces so far .......\n")
# %%
