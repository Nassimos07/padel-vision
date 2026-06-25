"""
Route B: tracking-data -> per-frame SVG -> rasterize -> composite -> HQ mp4.
Plug your tracker output into `tracks_for_frame(i)`. Everything else is reusable.
Animations are computed as f(time) so frames are deterministic & seekable.
"""
import math, io, cairosvg, imageio.v2 as imageio
from PIL import Image

W, H, FPS = 1463, 812, 30

# ---- your data goes here -------------------------------------------------
# Per player per frame: feet/head in image px, plus metrics. In production this
# comes from your tracker (e.g. ByteTrack ids + court-homography foot point).
BASE = [
 dict(id='P1',color='#FF3D8B',feet=(468,792),head=(455,648),rx=74,ry=28,speed=18.4,state='SPRINT',move=(-0.55,-1)),
 dict(id='P2',color='#FFB020',feet=(958,778),head=(958,636),rx=74,ry=28,speed=12.1,state='COVER', move=(0.35,-1)),
 dict(id='P3',color='#27E08A',feet=(646,332),head=(646,232),rx=46,ry=16,speed=6.7, state='NET',   move=None),
 dict(id='P4',color='#2E9BFF',feet=(800,252),head=(800,150),rx=42,ry=15,speed=9.3, state='STRIKE',move=(0.2,1)),
]
def tracks_for_frame(i):
    """Return players for frame i. Demo: tiny idle sway so a still feels alive."""
    t = i / FPS
    out = []
    for k, p in enumerate(BASE):
        q = dict(p)
        sway = math.sin(2*math.pi*t/3.4 + k) * 1.6      # ±1.6 px idle motion
        fx, fy = p['feet']; hx, hy = p['head']
        q['feet'] = (fx + sway, fy)
        q['head'] = (hx + sway*0.8, hy)
        q['speed'] = f"{p['speed'] + 1.2*math.sin(2*math.pi*t/2.0 + k):.1f}"  # jitter readout
        out.append(q)
    return out

def ell_len(a,b): return math.pi*(3*(a+b)-math.sqrt((3*a+b)*(a+3*b)))

def build_svg(players, t):
    S=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<defs>']
    for p in players:
        c=p['color']
        S.append(f'<radialGradient id="fl-{p["id"]}" cx="50%" cy="50%" r="50%">'
                 f'<stop offset="0%" stop-color="{c}" stop-opacity="0"/>'
                 f'<stop offset="62%" stop-color="{c}" stop-opacity="0.10"/>'
                 f'<stop offset="100%" stop-color="{c}" stop-opacity="0.34"/></radialGradient>'
                 f'<filter id="gl-{p["id"]}" x="-60%" y="-60%" width="220%" height="220%">'
                 f'<feGaussianBlur stdDeviation="5"/></filter>')
    S.append('</defs>')
    for k,p in enumerate(players):
        fx,fy=p['feet']; hx,hy=p['head']; c=p['color']; rx,ry=p['rx'],p['ry']
        # entrance: ease-out over 0.55s, staggered
        a=max(0.0,min(1.0,(t-(0.25+k*0.12))/0.55)); a=1-(1-a)**3
        if a<=0: continue
        oy=(1-a)*6
        S.append(f'<g opacity="{a:.3f}" transform="translate(0 {oy:.2f})">')
        S.append(f'<line x1="{fx:.1f}" y1="{fy}" x2="{hx:.1f}" y2="{hy+8}" stroke="{c}" stroke-width="1.5" stroke-dasharray="1 7" stroke-linecap="round" opacity="0.45"/>')
        S.append(f'<ellipse cx="{fx:.1f}" cy="{fy}" rx="{rx}" ry="{ry}" fill="url(#fl-{p["id"]})"/>')
        pulse=0.30+0.30*(0.5+0.5*math.sin(2*math.pi*t/2.6+k))
        S.append(f'<ellipse cx="{fx:.1f}" cy="{fy}" rx="{rx}" ry="{ry}" fill="none" stroke="{c}" stroke-width="5" filter="url(#gl-{p["id"]})" opacity="{pulse:.3f}"/>')
        S.append(f'<ellipse cx="{fx:.1f}" cy="{fy}" rx="{rx}" ry="{ry}" fill="none" stroke="{c}" stroke-width="2" opacity="0.9"/>')
        for tk in range(12):
            ang=tk/12*2*math.pi
            x1=fx+math.cos(ang)*rx; y1=fy+math.sin(ang)*ry
            x2=fx+math.cos(ang)*(rx+abs(math.cos(ang))*6+3); y2=fy+math.sin(ang)*(ry+abs(math.sin(ang))*3+1.5)
            S.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="1" opacity="0.35"/>')
        L=ell_len(rx,ry); seg=L*0.16; off=-(t/3.0)*L
        S.append(f'<ellipse cx="{fx:.1f}" cy="{fy}" rx="{rx}" ry="{ry}" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-dasharray="{seg:.1f} {L-seg:.1f}" stroke-dashoffset="{off:.1f}" opacity="0.85"/>')
        S.append(f'<circle cx="{fx:.1f}" cy="{fy}" r="2" fill="{c}"/>')
        if p['move']:
            m=p['move']; ml=math.hypot(*m); ux,uy=m[0]/ml,m[1]/ml
            ex=fx+ux*rx*0.95; ey=fy+uy*ry*1.7; aa=math.atan2(ey-fy,ex-fx); ah=7
            S.append(f'<line x1="{fx:.1f}" y1="{fy}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{c}" stroke-width="2.2" stroke-linecap="round" opacity="0.8"/>')
            S.append(f'<path d="M{ex:.1f},{ey:.1f} L{ex-math.cos(aa-0.5)*ah:.1f},{ey-math.sin(aa-0.5)*ah:.1f} L{ex-math.cos(aa+0.5)*ah:.1f},{ey-math.sin(aa+0.5)*ah:.1f} Z" fill="{c}" opacity="0.85"/>')
        S.append(f'<circle cx="{hx:.1f}" cy="{hy+8}" r="7" fill="none" stroke="{c}" stroke-width="1.5" opacity="0.85"/>')
        S.append(f'<circle cx="{hx:.1f}" cy="{hy+8}" r="2.5" fill="{c}"/>')
        cw,ch=156,54; cx=hx-cw/2; cyB=hy-20; cy=cyB-ch
        S.append(f'<line x1="{hx:.1f}" y1="{cyB+10}" x2="{hx:.1f}" y2="{hy+8}" stroke="{c}" stroke-width="1.5" opacity="0.7"/>')
        S.append(f'<path d="M{hx-7:.1f},{cyB} L{hx+7:.1f},{cyB} L{hx:.1f},{cyB+10} Z" fill="#0b1018" fill-opacity="0.92"/>')
        S.append(f'<rect x="{cx:.1f}" y="{cy}" width="{cw}" height="{ch}" rx="9" fill="#0b1018" fill-opacity="0.86" stroke="#fff" stroke-opacity="0.12"/>')
        S.append(f'<rect x="{cx:.1f}" y="{cy+1}" width="4" height="{ch-2}" rx="2" fill="{c}"/>')
        S.append(f'<text x="{cx+18:.1f}" y="{cy+35}" fill="#fff" font-family="monospace" font-size="24" font-weight="700">{p["id"]}</text>')
        S.append(f'<line x1="{cx+66:.1f}" y1="{cy+10}" x2="{cx+66:.1f}" y2="{cy+ch-10}" stroke="#fff" stroke-opacity="0.12"/>')
        S.append(f'<text x="{cx+cw-14:.1f}" y="{cy+22}" text-anchor="end" fill="{c}" font-family="sans-serif" font-size="11" font-weight="700">{p["state"]}</text>')
        S.append(f'<text x="{cx+cw-14:.1f}" y="{cy+43}" text-anchor="end" fill="#fff" font-family="monospace" font-size="17" font-weight="700">{p["speed"]}<tspan fill="#fff" fill-opacity="0.5" font-size="10"> km/h</tspan></text>')
        S.append('</g>')
    # HUD with live clock + pulsing dot
    frac=(t%1.6)/1.6; dotop=0.3+0.7*(0.5+0.5*math.cos(2*math.pi*frac))
    ping_r=5+8*frac; ping_op=0.7*(1-frac)
    mm=int(t)//60; ss=t%60
    S.append('<rect x="20" y="20" width="316" height="66" rx="12" fill="#0b1018" fill-opacity="0.72" stroke="#fff" stroke-opacity="0.14"/>')
    S.append(f'<circle cx="42" cy="44" r="{ping_r:.1f}" fill="none" stroke="#37e0c8" stroke-width="1.5" opacity="{ping_op:.3f}"/>')
    S.append(f'<circle cx="42" cy="44" r="5" fill="#37e0c8" opacity="{dotop:.3f}"/>')
    S.append('<text x="58" y="40" fill="#fff" font-family="monospace" font-size="15" font-weight="700" letter-spacing="2.5">PADEL ANALYTICS</text>')
    S.append('<text x="58" y="62" fill="#fff" fill-opacity="0.55" font-family="sans-serif" font-size="11" font-weight="600" letter-spacing="2">LIVE TRACKING</text>')
    S.append(f'<text x="336" y="40" text-anchor="end" fill="#fff" font-family="monospace" font-size="14">{mm:02d}:{ss:04.1f}</text>')
    S.append('<text x="336" y="62" text-anchor="end" fill="#37e0c8" font-family="monospace" font-size="12">4 / 4 LOCKED</text>')
    S.append('</svg>')
    return ''.join(S)

def main(out='padel_ar_demo.mp4', seconds=4):
    base = Image.open('/mnt/user-data/uploads/1781701001925_image.png').convert('RGBA')
    n = int(seconds*FPS)
    w = imageio.get_writer(out, fps=FPS, codec='libx264',
                           macro_block_size=1, ffmpeg_params=['-pix_fmt','yuv420p','-crf','16'])
    for i in range(n):
        t = i/FPS
        svg = build_svg(tracks_for_frame(i), t)
        png = cairosvg.svg2png(bytestring=svg.encode(), output_width=W, output_height=H)
        ov = Image.open(io.BytesIO(png)).convert('RGBA')
        frame = base.copy(); frame.alpha_composite(ov)
        if frame.width % 2: frame = frame.resize((frame.width+1, frame.height))
        w.append_data(np_rgb(frame))
        if i % 30 == 0: print(f'frame {i}/{n}')
    w.close(); print('wrote', out)

def np_rgb(img):
    import numpy as np
    return np.asarray(img.convert('RGB'))

if __name__ == '__main__':
    main()
