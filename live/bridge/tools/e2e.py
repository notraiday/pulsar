# fake station: synthesises a question with the TTS, feeds it as 16 kHz mic audio, measures the reply
import asyncio, audioop, os, struct, time, aiohttp
K=os.environ["OPENROUTER_API_KEY"]; P=os.environ.get("LIVE_OR_PROXY")
def fr(t,p=b""): return t+struct.pack("<I",len(p))+p
async def main():
    async with aiohttp.ClientSession() as h:
        r=await h.post("https://openrouter.ai/api/v1/audio/speech",headers={"Authorization":"Bearer "+K},proxy=P,json={"model":"google/gemini-3.8-flash-tts","input":os.environ.get("Q","Какая сейчас температура в гостиной?"),"voice":"Puck","response_format":"pcm","provider":{"zdr":True,"data_collection":"deny"}})
        pcm24=await r.read()
    pcm16,_=audioop.ratecv(pcm24,2,1,24000,16000,None)
    rd,wr=await asyncio.open_connection("127.0.0.1",int(os.environ.get("PORT","10700"))); wr.write(fr(b"H",b"{}"))
    sig=b"\0"*16000+pcm16+b"\0"*(32000*30); t_end=None; t0=time.monotonic(); audio=0; first=None; last_audio=None
    async def mic():
        nonlocal t_end
        for i in range(0,len(sig),1024):
            if t_end is None and i>=16000+len(pcm16): t_end=time.monotonic()
            wr.write(fr(b"A",sig[i:i+1024])); await wr.drain(); await asyncio.sleep(0.032)
    asyncio.create_task(mic())
    while time.monotonic()-t0<45:
        try: hd=await asyncio.wait_for(rd.readexactly(5),12)
        except Exception: break
        t,n=hd[:1],struct.unpack("<I",hd[1:])[0]; p=await rd.readexactly(n) if n else b""
        now=time.monotonic()
        if t==b"A":
            if t_end and (last_audio is None or now-last_audio>0.35): print(f"  audio burst starts {now-t_end:5.2f}s after the question ended")
            audio+=len(p); last_audio=now
            if first is None and t_end: first=now-t_end; print(f"first reply audio {first:.2f}s after the question ended")
        elif t==b"S": print(f"{now-t0:5.1f}s state {p.decode()}")
        elif t==b"X": print(f"{now-t0:5.1f}s end: {p.decode()}"); break
        if t==b"A" and audio>0: asyncio.get_event_loop().call_later(0, lambda: None)
    print(f"reply audio {audio/48000:.1f}s")
    # tell the bridge the audio has been played
asyncio.run(main())
