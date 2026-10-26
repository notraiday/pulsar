# fake station: question 1, then the owner talks over the answer with question 2
import asyncio, audioop, os, struct, time, aiohttp
K=os.environ["OPENROUTER_API_KEY"]; P=os.environ.get("LIVE_OR_PROXY")
GAP=float(os.environ.get("GAP","2.4")); Q1=os.environ.get("Q1","Какое у меня расписание на завтра?"); Q2=os.environ.get("Q2","И погоду тоже.")
def fr(t,p=b""): return t+struct.pack("<I",len(p))+p
async def tts16(h,t):
    r=await h.post("https://openrouter.ai/api/v1/audio/speech",headers={"Authorization":"Bearer "+K},proxy=P,json={"model":"elevenlabs/eleven-turbo-v2.5","input":t,"voice":"george","response_format":"pcm"})
    pcm,_=audioop.ratecv(await r.read(),2,1,24000,16000,None); return pcm
async def main():
    async with aiohttp.ClientSession() as h:
        a=await tts16(h,Q1); b=await tts16(h,Q2)
    sig=b"\0"*16000+a+b"\0"*int(32000*GAP)+b+b"\0"*(32000*30)
    t_q1_end=16000+len(a)
    rd,wr=await asyncio.open_connection("127.0.0.1",int(os.environ.get("PORT","10701"))); wr.write(fr(b"H",b"{}"))
    t0=time.monotonic()
    async def mic():
        for i in range(0,len(sig),1024):
            wr.write(fr(b"A",sig[i:i+1024])); await wr.drain(); await asyncio.sleep(0.032)
    asyncio.create_task(mic())
    last=None; marks={"q1 ends":t_q1_end/32000,"q2 starts":(t_q1_end+int(32000*GAP))/32000,"q2 ends":(t_q1_end+int(32000*GAP)+len(b))/32000}
    print("fake station timeline (s since start):",{k:round(v,1) for k,v in marks.items()})
    while time.monotonic()-t0<45:
        try: hd=await asyncio.wait_for(rd.readexactly(5),14)
        except Exception: break
        t,n=hd[:1],struct.unpack("<I",hd[1:])[0]; p=await rd.readexactly(n) if n else b""
        now=time.monotonic()-t0
        if t==b"A":
            if last is None or now-last>0.5: print(f"  {now:5.1f}s audio burst starts")
            last=now
        elif t==b"F": print(f"  {now:5.1f}s FLUSH (barge-in)")
        elif t==b"S": print(f"  {now:5.1f}s state {p.decode()}")
        elif t==b"X": print(f"  {now:5.1f}s end: {p.decode()}"); break
asyncio.run(main())
