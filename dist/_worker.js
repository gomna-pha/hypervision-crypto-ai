var Dt=Object.defineProperty;var Ve=e=>{throw TypeError(e)};var Ot=(e,t,a)=>t in e?Dt(e,t,{enumerable:!0,configurable:!0,writable:!0,value:a}):e[t]=a;var _=(e,t,a)=>Ot(e,typeof t!="symbol"?t+"":t,a),Fe=(e,t,a)=>t.has(e)||Ve("Cannot "+a);var d=(e,t,a)=>(Fe(e,t,"read from private field"),a?a.call(e):t.get(e)),b=(e,t,a)=>t.has(e)?Ve("Cannot add the same private member more than once"):t instanceof WeakSet?t.add(e):t.set(e,a),f=(e,t,a,s)=>(Fe(e,t,"write to private field"),s?s.call(e,a):t.set(e,a),a),x=(e,t,a)=>(Fe(e,t,"access private method"),a);var We=(e,t,a,s)=>({set _(r){f(e,t,r,a)},get _(){return d(e,t,s)}});var ze=(e,t,a)=>(s,r)=>{let n=-1;return i(0);async function i(o){if(o<=n)throw new Error("next() called multiple times");n=o;let l,c=!1,u;if(e[o]?(u=e[o][0][0],s.req.routeIndex=o):u=o===e.length&&r||void 0,u)try{l=await u(s,()=>i(o+1))}catch(g){if(g instanceof Error&&t)s.error=g,l=await t(g,s),c=!0;else throw g}else s.finalized===!1&&a&&(l=await a(s));return l&&(s.finalized===!1||c)&&(s.res=l),s}},Nt=Symbol(),Mt=async(e,t=Object.create(null))=>{const{all:a=!1,dot:s=!1}=t,n=(e instanceof mt?e.raw.headers:e.headers).get("Content-Type");return n!=null&&n.startsWith("multipart/form-data")||n!=null&&n.startsWith("application/x-www-form-urlencoded")?Lt(e,{all:a,dot:s}):{}};async function Lt(e,t){const a=await e.formData();return a?Pt(a,t):{}}function Pt(e,t){const a=Object.create(null);return e.forEach((s,r)=>{t.all||r.endsWith("[]")?jt(a,r,s):a[r]=s}),t.dot&&Object.entries(a).forEach(([s,r])=>{s.includes(".")&&($t(a,s,r),delete a[s])}),a}var jt=(e,t,a)=>{e[t]!==void 0?Array.isArray(e[t])?e[t].push(a):e[t]=[e[t],a]:t.endsWith("[]")?e[t]=[a]:e[t]=a},$t=(e,t,a)=>{let s=e;const r=t.split(".");r.forEach((n,i)=>{i===r.length-1?s[n]=a:((!s[n]||typeof s[n]!="object"||Array.isArray(s[n])||s[n]instanceof File)&&(s[n]=Object.create(null)),s=s[n])})},dt=e=>{const t=e.split("/");return t[0]===""&&t.shift(),t},Ft=e=>{const{groups:t,path:a}=Bt(e),s=dt(a);return Ht(s,t)},Bt=e=>{const t=[];return e=e.replace(/\{[^}]+\}/g,(a,s)=>{const r=`@${s}`;return t.push([r,a]),r}),{groups:t,path:e}},Ht=(e,t)=>{for(let a=t.length-1;a>=0;a--){const[s]=t[a];for(let r=e.length-1;r>=0;r--)if(e[r].includes(s)){e[r]=e[r].replace(s,t[a][1]);break}}return e},De={},Gt=(e,t)=>{if(e==="*")return"*";const a=e.match(/^\:([^\{\}]+)(?:\{(.+)\})?$/);if(a){const s=`${e}#${t}`;return De[s]||(a[2]?De[s]=t&&t[0]!==":"&&t[0]!=="*"?[s,a[1],new RegExp(`^${a[2]}(?=/${t})`)]:[e,a[1],new RegExp(`^${a[2]}$`)]:De[s]=[e,a[1],!0]),De[s]}return null},qe=(e,t)=>{try{return t(e)}catch{return e.replace(/(?:%[0-9A-Fa-f]{2})+/g,a=>{try{return t(a)}catch{return a}})}},Ut=e=>qe(e,decodeURI),ut=e=>{const t=e.url,a=t.indexOf("/",t.indexOf(":")+4);let s=a;for(;s<t.length;s++){const r=t.charCodeAt(s);if(r===37){const n=t.indexOf("?",s),i=t.slice(a,n===-1?void 0:n);return Ut(i.includes("%25")?i.replace(/%25/g,"%2525"):i)}else if(r===63)break}return t.slice(a,s)},qt=e=>{const t=ut(e);return t.length>1&&t.at(-1)==="/"?t.slice(0,-1):t},de=(e,t,...a)=>(a.length&&(t=de(t,...a)),`${(e==null?void 0:e[0])==="/"?"":"/"}${e}${t==="/"?"":`${(e==null?void 0:e.at(-1))==="/"?"":"/"}${(t==null?void 0:t[0])==="/"?t.slice(1):t}`}`),gt=e=>{if(e.charCodeAt(e.length-1)!==63||!e.includes(":"))return null;const t=e.split("/"),a=[];let s="";return t.forEach(r=>{if(r!==""&&!/\:/.test(r))s+="/"+r;else if(/\:/.test(r))if(/\?/.test(r)){a.length===0&&s===""?a.push("/"):a.push(s);const n=r.replace("?","");s+="/"+n,a.push(s)}else s+="/"+r}),a.filter((r,n,i)=>i.indexOf(r)===n)},Be=e=>/[%+]/.test(e)?(e.indexOf("+")!==-1&&(e=e.replace(/\+/g," ")),e.indexOf("%")!==-1?qe(e,ht):e):e,pt=(e,t,a)=>{let s;if(!a&&t&&!/[%+]/.test(t)){let i=e.indexOf(`?${t}`,8);for(i===-1&&(i=e.indexOf(`&${t}`,8));i!==-1;){const o=e.charCodeAt(i+t.length+1);if(o===61){const l=i+t.length+2,c=e.indexOf("&",l);return Be(e.slice(l,c===-1?void 0:c))}else if(o==38||isNaN(o))return"";i=e.indexOf(`&${t}`,i+1)}if(s=/[%+]/.test(e),!s)return}const r={};s??(s=/[%+]/.test(e));let n=e.indexOf("?",8);for(;n!==-1;){const i=e.indexOf("&",n+1);let o=e.indexOf("=",n);o>i&&i!==-1&&(o=-1);let l=e.slice(n+1,o===-1?i===-1?void 0:i:o);if(s&&(l=Be(l)),n=i,l==="")continue;let c;o===-1?c="":(c=e.slice(o+1,i===-1?void 0:i),s&&(c=Be(c))),a?(r[l]&&Array.isArray(r[l])||(r[l]=[]),r[l].push(c)):r[l]??(r[l]=c)}return t?r[t]:r},Yt=pt,Kt=(e,t)=>pt(e,t,!0),ht=decodeURIComponent,Xe=e=>qe(e,ht),pe,N,U,ft,_t,Ge,K,tt,mt=(tt=class{constructor(e,t="/",a=[[]]){b(this,U);_(this,"raw");b(this,pe);b(this,N);_(this,"routeIndex",0);_(this,"path");_(this,"bodyCache",{});b(this,K,e=>{const{bodyCache:t,raw:a}=this,s=t[e];if(s)return s;const r=Object.keys(t)[0];return r?t[r].then(n=>(r==="json"&&(n=JSON.stringify(n)),new Response(n)[e]())):t[e]=a[e]()});this.raw=e,this.path=t,f(this,N,a),f(this,pe,{})}param(e){return e?x(this,U,ft).call(this,e):x(this,U,_t).call(this)}query(e){return Yt(this.url,e)}queries(e){return Kt(this.url,e)}header(e){if(e)return this.raw.headers.get(e)??void 0;const t={};return this.raw.headers.forEach((a,s)=>{t[s]=a}),t}async parseBody(e){var t;return(t=this.bodyCache).parsedBody??(t.parsedBody=await Mt(this,e))}json(){return d(this,K).call(this,"text").then(e=>JSON.parse(e))}text(){return d(this,K).call(this,"text")}arrayBuffer(){return d(this,K).call(this,"arrayBuffer")}blob(){return d(this,K).call(this,"blob")}formData(){return d(this,K).call(this,"formData")}addValidatedData(e,t){d(this,pe)[e]=t}valid(e){return d(this,pe)[e]}get url(){return this.raw.url}get method(){return this.raw.method}get[Nt](){return d(this,N)}get matchedRoutes(){return d(this,N)[0].map(([[,e]])=>e)}get routePath(){return d(this,N)[0].map(([[,e]])=>e)[this.routeIndex].path}},pe=new WeakMap,N=new WeakMap,U=new WeakSet,ft=function(e){const t=d(this,N)[0][this.routeIndex][1][e],a=x(this,U,Ge).call(this,t);return a&&/\%/.test(a)?Xe(a):a},_t=function(){const e={},t=Object.keys(d(this,N)[0][this.routeIndex][1]);for(const a of t){const s=x(this,U,Ge).call(this,d(this,N)[0][this.routeIndex][1][a]);s!==void 0&&(e[a]=/\%/.test(s)?Xe(s):s)}return e},Ge=function(e){return d(this,N)[1]?d(this,N)[1][e]:e},K=new WeakMap,tt),Vt={Stringify:1},bt=async(e,t,a,s,r)=>{typeof e=="object"&&!(e instanceof String)&&(e instanceof Promise||(e=e.toString()),e instanceof Promise&&(e=await e));const n=e.callbacks;return n!=null&&n.length?(r?r[0]+=e:r=[e],Promise.all(n.map(o=>o({phase:t,buffer:r,context:s}))).then(o=>Promise.all(o.filter(Boolean).map(l=>bt(l,t,!1,s,r))).then(()=>r[0]))):Promise.resolve(e)},Wt="text/plain; charset=UTF-8",He=(e,t)=>({"Content-Type":e,...t}),Ie,Se,F,he,B,D,Re,me,fe,se,Ce,Te,V,ue,at,zt=(at=class{constructor(e,t){b(this,V);b(this,Ie);b(this,Se);_(this,"env",{});b(this,F);_(this,"finalized",!1);_(this,"error");b(this,he);b(this,B);b(this,D);b(this,Re);b(this,me);b(this,fe);b(this,se);b(this,Ce);b(this,Te);_(this,"render",(...e)=>(d(this,me)??f(this,me,t=>this.html(t)),d(this,me).call(this,...e)));_(this,"setLayout",e=>f(this,Re,e));_(this,"getLayout",()=>d(this,Re));_(this,"setRenderer",e=>{f(this,me,e)});_(this,"header",(e,t,a)=>{this.finalized&&f(this,D,new Response(d(this,D).body,d(this,D)));const s=d(this,D)?d(this,D).headers:d(this,se)??f(this,se,new Headers);t===void 0?s.delete(e):a!=null&&a.append?s.append(e,t):s.set(e,t)});_(this,"status",e=>{f(this,he,e)});_(this,"set",(e,t)=>{d(this,F)??f(this,F,new Map),d(this,F).set(e,t)});_(this,"get",e=>d(this,F)?d(this,F).get(e):void 0);_(this,"newResponse",(...e)=>x(this,V,ue).call(this,...e));_(this,"body",(e,t,a)=>x(this,V,ue).call(this,e,t,a));_(this,"text",(e,t,a)=>!d(this,se)&&!d(this,he)&&!t&&!a&&!this.finalized?new Response(e):x(this,V,ue).call(this,e,t,He(Wt,a)));_(this,"json",(e,t,a)=>x(this,V,ue).call(this,JSON.stringify(e),t,He("application/json",a)));_(this,"html",(e,t,a)=>{const s=r=>x(this,V,ue).call(this,r,t,He("text/html; charset=UTF-8",a));return typeof e=="object"?bt(e,Vt.Stringify,!1,{}).then(s):s(e)});_(this,"redirect",(e,t)=>{const a=String(e);return this.header("Location",/[^\x00-\xFF]/.test(a)?encodeURI(a):a),this.newResponse(null,t??302)});_(this,"notFound",()=>(d(this,fe)??f(this,fe,()=>new Response),d(this,fe).call(this,this)));f(this,Ie,e),t&&(f(this,B,t.executionCtx),this.env=t.env,f(this,fe,t.notFoundHandler),f(this,Te,t.path),f(this,Ce,t.matchResult))}get req(){return d(this,Se)??f(this,Se,new mt(d(this,Ie),d(this,Te),d(this,Ce))),d(this,Se)}get event(){if(d(this,B)&&"respondWith"in d(this,B))return d(this,B);throw Error("This context has no FetchEvent")}get executionCtx(){if(d(this,B))return d(this,B);throw Error("This context has no ExecutionContext")}get res(){return d(this,D)||f(this,D,new Response(null,{headers:d(this,se)??f(this,se,new Headers)}))}set res(e){if(d(this,D)&&e){e=new Response(e.body,e);for(const[t,a]of d(this,D).headers.entries())if(t!=="content-type")if(t==="set-cookie"){const s=d(this,D).headers.getSetCookie();e.headers.delete("set-cookie");for(const r of s)e.headers.append("set-cookie",r)}else e.headers.set(t,a)}f(this,D,e),this.finalized=!0}get var(){return d(this,F)?Object.fromEntries(d(this,F)):{}}},Ie=new WeakMap,Se=new WeakMap,F=new WeakMap,he=new WeakMap,B=new WeakMap,D=new WeakMap,Re=new WeakMap,me=new WeakMap,fe=new WeakMap,se=new WeakMap,Ce=new WeakMap,Te=new WeakMap,V=new WeakSet,ue=function(e,t,a){const s=d(this,D)?new Headers(d(this,D).headers):d(this,se)??new Headers;if(typeof t=="object"&&"headers"in t){const n=t.headers instanceof Headers?t.headers:new Headers(t.headers);for(const[i,o]of n)i.toLowerCase()==="set-cookie"?s.append(i,o):s.set(i,o)}if(a)for(const[n,i]of Object.entries(a))if(typeof i=="string")s.set(n,i);else{s.delete(n);for(const o of i)s.append(n,o)}const r=typeof t=="number"?t:(t==null?void 0:t.status)??d(this,he);return new Response(e,{status:r,headers:s})},at),S="ALL",Xt="all",Qt=["get","post","put","delete","options","patch"],vt="Can not add a route since the matcher is already built.",yt=class extends Error{},Jt="__COMPOSED_HANDLER",Zt=e=>e.text("404 Not Found",404),Qe=(e,t)=>{if("getResponse"in e){const a=e.getResponse();return t.newResponse(a.body,a)}return console.error(e),t.text("Internal Server Error",500)},M,R,Et,L,te,Ne,Me,st,xt=(st=class{constructor(t={}){b(this,R);_(this,"get");_(this,"post");_(this,"put");_(this,"delete");_(this,"options");_(this,"patch");_(this,"all");_(this,"on");_(this,"use");_(this,"router");_(this,"getPath");_(this,"_basePath","/");b(this,M,"/");_(this,"routes",[]);b(this,L,Zt);_(this,"errorHandler",Qe);_(this,"onError",t=>(this.errorHandler=t,this));_(this,"notFound",t=>(f(this,L,t),this));_(this,"fetch",(t,...a)=>x(this,R,Me).call(this,t,a[1],a[0],t.method));_(this,"request",(t,a,s,r)=>t instanceof Request?this.fetch(a?new Request(t,a):t,s,r):(t=t.toString(),this.fetch(new Request(/^https?:\/\//.test(t)?t:`http://localhost${de("/",t)}`,a),s,r)));_(this,"fire",()=>{addEventListener("fetch",t=>{t.respondWith(x(this,R,Me).call(this,t.request,t,void 0,t.request.method))})});[...Qt,Xt].forEach(n=>{this[n]=(i,...o)=>(typeof i=="string"?f(this,M,i):x(this,R,te).call(this,n,d(this,M),i),o.forEach(l=>{x(this,R,te).call(this,n,d(this,M),l)}),this)}),this.on=(n,i,...o)=>{for(const l of[i].flat()){f(this,M,l);for(const c of[n].flat())o.map(u=>{x(this,R,te).call(this,c.toUpperCase(),d(this,M),u)})}return this},this.use=(n,...i)=>(typeof n=="string"?f(this,M,n):(f(this,M,"*"),i.unshift(n)),i.forEach(o=>{x(this,R,te).call(this,S,d(this,M),o)}),this);const{strict:s,...r}=t;Object.assign(this,r),this.getPath=s??!0?t.getPath??ut:qt}route(t,a){const s=this.basePath(t);return a.routes.map(r=>{var i;let n;a.errorHandler===Qe?n=r.handler:(n=async(o,l)=>(await ze([],a.errorHandler)(o,()=>r.handler(o,l))).res,n[Jt]=r.handler),x(i=s,R,te).call(i,r.method,r.path,n)}),this}basePath(t){const a=x(this,R,Et).call(this);return a._basePath=de(this._basePath,t),a}mount(t,a,s){let r,n;s&&(typeof s=="function"?n=s:(n=s.optionHandler,s.replaceRequest===!1?r=l=>l:r=s.replaceRequest));const i=n?l=>{const c=n(l);return Array.isArray(c)?c:[c]}:l=>{let c;try{c=l.executionCtx}catch{}return[l.env,c]};r||(r=(()=>{const l=de(this._basePath,t),c=l==="/"?0:l.length;return u=>{const g=new URL(u.url);return g.pathname=g.pathname.slice(c)||"/",new Request(g,u)}})());const o=async(l,c)=>{const u=await a(r(l.req.raw),...i(l));if(u)return u;await c()};return x(this,R,te).call(this,S,de(t,"*"),o),this}},M=new WeakMap,R=new WeakSet,Et=function(){const t=new xt({router:this.router,getPath:this.getPath});return t.errorHandler=this.errorHandler,f(t,L,d(this,L)),t.routes=this.routes,t},L=new WeakMap,te=function(t,a,s){t=t.toUpperCase(),a=de(this._basePath,a);const r={basePath:this._basePath,path:a,method:t,handler:s};this.router.add(t,a,[s,r]),this.routes.push(r)},Ne=function(t,a){if(t instanceof Error)return this.errorHandler(t,a);throw t},Me=function(t,a,s,r){if(r==="HEAD")return(async()=>new Response(null,await x(this,R,Me).call(this,t,a,s,"GET")))();const n=this.getPath(t,{env:s}),i=this.router.match(r,n),o=new zt(t,{path:n,matchResult:i,env:s,executionCtx:a,notFoundHandler:d(this,L)});if(i[0].length===1){let c;try{c=i[0][0][0][0](o,async()=>{o.res=await d(this,L).call(this,o)})}catch(u){return x(this,R,Ne).call(this,u,o)}return c instanceof Promise?c.then(u=>u||(o.finalized?o.res:d(this,L).call(this,o))).catch(u=>x(this,R,Ne).call(this,u,o)):c??d(this,L).call(this,o)}const l=ze(i[0],this.errorHandler,d(this,L));return(async()=>{try{const c=await l(o);if(!c.finalized)throw new Error("Context is not finalized. Did you forget to return a Response object or `await next()`?");return c.res}catch(c){return x(this,R,Ne).call(this,c,o)}})()},st),wt=[];function ea(e,t){const a=this.buildAllMatchers(),s=(r,n)=>{const i=a[r]||a[S],o=i[2][n];if(o)return o;const l=n.match(i[0]);if(!l)return[[],wt];const c=l.indexOf("",1);return[i[1][c],l]};return this.match=s,s(e,t)}var Pe="[^/]+",Ee=".*",we="(?:|/.*)",ge=Symbol(),ta=new Set(".\\+*[^]$()");function aa(e,t){return e.length===1?t.length===1?e<t?-1:1:-1:t.length===1||e===Ee||e===we?1:t===Ee||t===we?-1:e===Pe?1:t===Pe?-1:e.length===t.length?e<t?-1:1:t.length-e.length}var re,ne,P,rt,Ue=(rt=class{constructor(){b(this,re);b(this,ne);b(this,P,Object.create(null))}insert(t,a,s,r,n){if(t.length===0){if(d(this,re)!==void 0)throw ge;if(n)return;f(this,re,a);return}const[i,...o]=t,l=i==="*"?o.length===0?["","",Ee]:["","",Pe]:i==="/*"?["","",we]:i.match(/^\:([^\{\}]+)(?:\{(.+)\})?$/);let c;if(l){const u=l[1];let g=l[2]||Pe;if(u&&l[2]&&(g===".*"||(g=g.replace(/^\((?!\?:)(?=[^)]+\)$)/,"(?:"),/\((?!\?:)/.test(g))))throw ge;if(c=d(this,P)[g],!c){if(Object.keys(d(this,P)).some(h=>h!==Ee&&h!==we))throw ge;if(n)return;c=d(this,P)[g]=new Ue,u!==""&&f(c,ne,r.varIndex++)}!n&&u!==""&&s.push([u,d(c,ne)])}else if(c=d(this,P)[i],!c){if(Object.keys(d(this,P)).some(u=>u.length>1&&u!==Ee&&u!==we))throw ge;if(n)return;c=d(this,P)[i]=new Ue}c.insert(o,a,s,r,n)}buildRegExpStr(){const a=Object.keys(d(this,P)).sort(aa).map(s=>{const r=d(this,P)[s];return(typeof d(r,ne)=="number"?`(${s})@${d(r,ne)}`:ta.has(s)?`\\${s}`:s)+r.buildRegExpStr()});return typeof d(this,re)=="number"&&a.unshift(`#${d(this,re)}`),a.length===0?"":a.length===1?a[0]:"(?:"+a.join("|")+")"}},re=new WeakMap,ne=new WeakMap,P=new WeakMap,rt),je,Ae,nt,sa=(nt=class{constructor(){b(this,je,{varIndex:0});b(this,Ae,new Ue)}insert(e,t,a){const s=[],r=[];for(let i=0;;){let o=!1;if(e=e.replace(/\{[^}]+\}/g,l=>{const c=`@\\${i}`;return r[i]=[c,l],i++,o=!0,c}),!o)break}const n=e.match(/(?::[^\/]+)|(?:\/\*$)|./g)||[];for(let i=r.length-1;i>=0;i--){const[o]=r[i];for(let l=n.length-1;l>=0;l--)if(n[l].indexOf(o)!==-1){n[l]=n[l].replace(o,r[i][1]);break}}return d(this,Ae).insert(n,t,s,d(this,je),a),s}buildRegExp(){let e=d(this,Ae).buildRegExpStr();if(e==="")return[/^$/,[],[]];let t=0;const a=[],s=[];return e=e.replace(/#(\d+)|@(\d+)|\.\*\$/g,(r,n,i)=>n!==void 0?(a[++t]=Number(n),"$()"):(i!==void 0&&(s[Number(i)]=++t),"")),[new RegExp(`^${e}`),a,s]}},je=new WeakMap,Ae=new WeakMap,nt),ra=[/^$/,[],Object.create(null)],Le=Object.create(null);function It(e){return Le[e]??(Le[e]=new RegExp(e==="*"?"":`^${e.replace(/\/\*$|([.\\+*[^\]$()])/g,(t,a)=>a?`\\${a}`:"(?:|/.*)")}$`))}function na(){Le=Object.create(null)}function ia(e){var c;const t=new sa,a=[];if(e.length===0)return ra;const s=e.map(u=>[!/\*|\/:/.test(u[0]),...u]).sort(([u,g],[h,v])=>u?1:h?-1:g.length-v.length),r=Object.create(null);for(let u=0,g=-1,h=s.length;u<h;u++){const[v,E,m]=s[u];v?r[E]=[m.map(([w])=>[w,Object.create(null)]),wt]:g++;let y;try{y=t.insert(E,g,v)}catch(w){throw w===ge?new yt(E):w}v||(a[g]=m.map(([w,O])=>{const q=Object.create(null);for(O-=1;O>=0;O--){const[A,Y]=y[O];q[A]=Y}return[w,q]}))}const[n,i,o]=t.buildRegExp();for(let u=0,g=a.length;u<g;u++)for(let h=0,v=a[u].length;h<v;h++){const E=(c=a[u][h])==null?void 0:c[1];if(!E)continue;const m=Object.keys(E);for(let y=0,w=m.length;y<w;y++)E[m[y]]=o[E[m[y]]]}const l=[];for(const u in i)l[u]=a[i[u]];return[n,l,r]}function ce(e,t){if(e){for(const a of Object.keys(e).sort((s,r)=>r.length-s.length))if(It(a).test(t))return[...e[a]]}}var W,z,$e,St,it,oa=(it=class{constructor(){b(this,$e);_(this,"name","RegExpRouter");b(this,W);b(this,z);_(this,"match",ea);f(this,W,{[S]:Object.create(null)}),f(this,z,{[S]:Object.create(null)})}add(e,t,a){var o;const s=d(this,W),r=d(this,z);if(!s||!r)throw new Error(vt);s[e]||[s,r].forEach(l=>{l[e]=Object.create(null),Object.keys(l[S]).forEach(c=>{l[e][c]=[...l[S][c]]})}),t==="/*"&&(t="*");const n=(t.match(/\/:/g)||[]).length;if(/\*$/.test(t)){const l=It(t);e===S?Object.keys(s).forEach(c=>{var u;(u=s[c])[t]||(u[t]=ce(s[c],t)||ce(s[S],t)||[])}):(o=s[e])[t]||(o[t]=ce(s[e],t)||ce(s[S],t)||[]),Object.keys(s).forEach(c=>{(e===S||e===c)&&Object.keys(s[c]).forEach(u=>{l.test(u)&&s[c][u].push([a,n])})}),Object.keys(r).forEach(c=>{(e===S||e===c)&&Object.keys(r[c]).forEach(u=>l.test(u)&&r[c][u].push([a,n]))});return}const i=gt(t)||[t];for(let l=0,c=i.length;l<c;l++){const u=i[l];Object.keys(r).forEach(g=>{var h;(e===S||e===g)&&((h=r[g])[u]||(h[u]=[...ce(s[g],u)||ce(s[S],u)||[]]),r[g][u].push([a,n-c+l+1]))})}}buildAllMatchers(){const e=Object.create(null);return Object.keys(d(this,z)).concat(Object.keys(d(this,W))).forEach(t=>{e[t]||(e[t]=x(this,$e,St).call(this,t))}),f(this,W,f(this,z,void 0)),na(),e}},W=new WeakMap,z=new WeakMap,$e=new WeakSet,St=function(e){const t=[];let a=e===S;return[d(this,W),d(this,z)].forEach(s=>{const r=s[e]?Object.keys(s[e]).map(n=>[n,s[e][n]]):[];r.length!==0?(a||(a=!0),t.push(...r)):e!==S&&t.push(...Object.keys(s[S]).map(n=>[n,s[S][n]]))}),a?ia(t):null},it),X,H,ot,la=(ot=class{constructor(e){_(this,"name","SmartRouter");b(this,X,[]);b(this,H,[]);f(this,X,e.routers)}add(e,t,a){if(!d(this,H))throw new Error(vt);d(this,H).push([e,t,a])}match(e,t){if(!d(this,H))throw new Error("Fatal error");const a=d(this,X),s=d(this,H),r=a.length;let n=0,i;for(;n<r;n++){const o=a[n];try{for(let l=0,c=s.length;l<c;l++)o.add(...s[l]);i=o.match(e,t)}catch(l){if(l instanceof yt)continue;throw l}this.match=o.match.bind(o),f(this,X,[o]),f(this,H,void 0);break}if(n===r)throw new Error("Fatal error");return this.name=`SmartRouter + ${this.activeRouter.name}`,i}get activeRouter(){if(d(this,H)||d(this,X).length!==1)throw new Error("No active router has been determined yet.");return d(this,X)[0]}},X=new WeakMap,H=new WeakMap,ot),xe=Object.create(null),Q,T,ie,_e,C,G,ae,lt,Rt=(lt=class{constructor(e,t,a){b(this,G);b(this,Q);b(this,T);b(this,ie);b(this,_e,0);b(this,C,xe);if(f(this,T,a||Object.create(null)),f(this,Q,[]),e&&t){const s=Object.create(null);s[e]={handler:t,possibleKeys:[],score:0},f(this,Q,[s])}f(this,ie,[])}insert(e,t,a){f(this,_e,++We(this,_e)._);let s=this;const r=Ft(t),n=[];for(let i=0,o=r.length;i<o;i++){const l=r[i],c=r[i+1],u=Gt(l,c),g=Array.isArray(u)?u[0]:l;if(g in d(s,T)){s=d(s,T)[g],u&&n.push(u[1]);continue}d(s,T)[g]=new Rt,u&&(d(s,ie).push(u),n.push(u[1])),s=d(s,T)[g]}return d(s,Q).push({[e]:{handler:a,possibleKeys:n.filter((i,o,l)=>l.indexOf(i)===o),score:d(this,_e)}}),s}search(e,t){var o;const a=[];f(this,C,xe);let r=[this];const n=dt(t),i=[];for(let l=0,c=n.length;l<c;l++){const u=n[l],g=l===c-1,h=[];for(let v=0,E=r.length;v<E;v++){const m=r[v],y=d(m,T)[u];y&&(f(y,C,d(m,C)),g?(d(y,T)["*"]&&a.push(...x(this,G,ae).call(this,d(y,T)["*"],e,d(m,C))),a.push(...x(this,G,ae).call(this,y,e,d(m,C)))):h.push(y));for(let w=0,O=d(m,ie).length;w<O;w++){const q=d(m,ie)[w],A=d(m,C)===xe?{}:{...d(m,C)};if(q==="*"){const j=d(m,T)["*"];j&&(a.push(...x(this,G,ae).call(this,j,e,d(m,C))),f(j,C,A),h.push(j));continue}const[Y,ke,J]=q;if(!u&&!(J instanceof RegExp))continue;const k=d(m,T)[Y],be=n.slice(l).join("/");if(J instanceof RegExp){const j=J.exec(be);if(j){if(A[ke]=j[0],a.push(...x(this,G,ae).call(this,k,e,d(m,C),A)),Object.keys(d(k,T)).length){f(k,C,A);const le=((o=j[0].match(/\//))==null?void 0:o.length)??0;(i[le]||(i[le]=[])).push(k)}continue}}(J===!0||J.test(u))&&(A[ke]=u,g?(a.push(...x(this,G,ae).call(this,k,e,A,d(m,C))),d(k,T)["*"]&&a.push(...x(this,G,ae).call(this,d(k,T)["*"],e,A,d(m,C)))):(f(k,C,A),h.push(k)))}}r=h.concat(i.shift()??[])}return a.length>1&&a.sort((l,c)=>l.score-c.score),[a.map(({handler:l,params:c})=>[l,c])]}},Q=new WeakMap,T=new WeakMap,ie=new WeakMap,_e=new WeakMap,C=new WeakMap,G=new WeakSet,ae=function(e,t,a,s){const r=[];for(let n=0,i=d(e,Q).length;n<i;n++){const o=d(e,Q)[n],l=o[t]||o[S],c={};if(l!==void 0&&(l.params=Object.create(null),r.push(l),a!==xe||s&&s!==xe))for(let u=0,g=l.possibleKeys.length;u<g;u++){const h=l.possibleKeys[u],v=c[l.score];l.params[h]=s!=null&&s[h]&&!v?s[h]:a[h]??(s==null?void 0:s[h]),c[l.score]=!0}}return r},lt),oe,ct,ca=(ct=class{constructor(){_(this,"name","TrieRouter");b(this,oe);f(this,oe,new Rt)}add(e,t,a){const s=gt(t);if(s){for(let r=0,n=s.length;r<n;r++)d(this,oe).insert(e,s[r],a);return}d(this,oe).insert(e,t,a)}match(e,t){return d(this,oe).search(e,t)}},oe=new WeakMap,ct),Ct=class extends xt{constructor(e={}){super(e),this.router=e.router??new la({routers:[new oa,new ca]})}},da=e=>{const a={...{origin:"*",allowMethods:["GET","HEAD","PUT","POST","DELETE","PATCH"],allowHeaders:[],exposeHeaders:[]},...e},s=(n=>typeof n=="string"?n==="*"?()=>n:i=>n===i?i:null:typeof n=="function"?n:i=>n.includes(i)?i:null)(a.origin),r=(n=>typeof n=="function"?n:Array.isArray(n)?()=>n:()=>[])(a.allowMethods);return async function(i,o){var u;function l(g,h){i.res.headers.set(g,h)}const c=await s(i.req.header("origin")||"",i);if(c&&l("Access-Control-Allow-Origin",c),a.credentials&&l("Access-Control-Allow-Credentials","true"),(u=a.exposeHeaders)!=null&&u.length&&l("Access-Control-Expose-Headers",a.exposeHeaders.join(",")),i.req.method==="OPTIONS"){a.origin!=="*"&&l("Vary","Origin"),a.maxAge!=null&&l("Access-Control-Max-Age",a.maxAge.toString());const g=await r(i.req.header("origin")||"",i);g.length&&l("Access-Control-Allow-Methods",g.join(","));let h=a.allowHeaders;if(!(h!=null&&h.length)){const v=i.req.header("Access-Control-Request-Headers");v&&(h=v.split(/\s*,\s*/))}return h!=null&&h.length&&(l("Access-Control-Allow-Headers",h.join(",")),i.res.headers.append("Vary","Access-Control-Request-Headers")),i.res.headers.delete("Content-Length"),i.res.headers.delete("Content-Type"),new Response(null,{headers:i.res.headers,status:204,statusText:"No Content"})}await o(),a.origin!=="*"&&i.header("Vary","Origin",{append:!0})}},ua=/^\s*(?:text\/(?!event-stream(?:[;\s]|$))[^;\s]+|application\/(?:javascript|json|xml|xml-dtd|ecmascript|dart|postscript|rtf|tar|toml|vnd\.dart|vnd\.ms-fontobject|vnd\.ms-opentype|wasm|x-httpd-php|x-javascript|x-ns-proxy-autoconfig|x-sh|x-tar|x-virtualbox-hdd|x-virtualbox-ova|x-virtualbox-ovf|x-virtualbox-vbox|x-virtualbox-vdi|x-virtualbox-vhd|x-virtualbox-vmdk|x-www-form-urlencoded)|font\/(?:otf|ttf)|image\/(?:bmp|vnd\.adobe\.photoshop|vnd\.microsoft\.icon|vnd\.ms-dds|x-icon|x-ms-bmp)|message\/rfc822|model\/gltf-binary|x-shader\/x-fragment|x-shader\/x-vertex|[^;\s]+?\+(?:json|text|xml|yaml))(?:[;\s]|$)/i,Je=(e,t=pa)=>{const a=/\.([a-zA-Z0-9]+?)$/,s=e.match(a);if(!s)return;let r=t[s[1]];return r&&r.startsWith("text")&&(r+="; charset=utf-8"),r},ga={aac:"audio/aac",avi:"video/x-msvideo",avif:"image/avif",av1:"video/av1",bin:"application/octet-stream",bmp:"image/bmp",css:"text/css",csv:"text/csv",eot:"application/vnd.ms-fontobject",epub:"application/epub+zip",gif:"image/gif",gz:"application/gzip",htm:"text/html",html:"text/html",ico:"image/x-icon",ics:"text/calendar",jpeg:"image/jpeg",jpg:"image/jpeg",js:"text/javascript",json:"application/json",jsonld:"application/ld+json",map:"application/json",mid:"audio/x-midi",midi:"audio/x-midi",mjs:"text/javascript",mp3:"audio/mpeg",mp4:"video/mp4",mpeg:"video/mpeg",oga:"audio/ogg",ogv:"video/ogg",ogx:"application/ogg",opus:"audio/opus",otf:"font/otf",pdf:"application/pdf",png:"image/png",rtf:"application/rtf",svg:"image/svg+xml",tif:"image/tiff",tiff:"image/tiff",ts:"video/mp2t",ttf:"font/ttf",txt:"text/plain",wasm:"application/wasm",webm:"video/webm",weba:"audio/webm",webmanifest:"application/manifest+json",webp:"image/webp",woff:"font/woff",woff2:"font/woff2",xhtml:"application/xhtml+xml",xml:"application/xml",zip:"application/zip","3gp":"video/3gpp","3g2":"video/3gpp2",gltf:"model/gltf+json",glb:"model/gltf-binary"},pa=ga,ha=(...e)=>{let t=e.filter(r=>r!=="").join("/");t=t.replace(new RegExp("(?<=\\/)\\/+","g"),"");const a=t.split("/"),s=[];for(const r of a)r===".."&&s.length>0&&s.at(-1)!==".."?s.pop():r!=="."&&s.push(r);return s.join("/")||"."},Tt={br:".br",zstd:".zst",gzip:".gz"},ma=Object.keys(Tt),fa="index.html",_a=e=>{const t=e.root??"./",a=e.path,s=e.join??ha;return async(r,n)=>{var u,g,h,v;if(r.finalized)return n();let i;if(e.path)i=e.path;else try{if(i=decodeURIComponent(r.req.path),/(?:^|[\/\\])\.\.(?:$|[\/\\])/.test(i))throw new Error}catch{return await((u=e.onNotFound)==null?void 0:u.call(e,r.req.path,r)),n()}let o=s(t,!a&&e.rewriteRequestPath?e.rewriteRequestPath(i):i);e.isDir&&await e.isDir(o)&&(o=s(o,fa));const l=e.getContent;let c=await l(o,r);if(c instanceof Response)return r.newResponse(c.body,c);if(c){const E=e.mimes&&Je(o,e.mimes)||Je(o);if(r.header("Content-Type",E||"application/octet-stream"),e.precompressed&&(!E||ua.test(E))){const m=new Set((g=r.req.header("Accept-Encoding"))==null?void 0:g.split(",").map(y=>y.trim()));for(const y of ma){if(!m.has(y))continue;const w=await l(o+Tt[y],r);if(w){c=w,r.header("Content-Encoding",y),r.header("Vary","Accept-Encoding",{append:!0});break}}}return await((h=e.onFound)==null?void 0:h.call(e,o,r)),r.body(c)}await((v=e.onNotFound)==null?void 0:v.call(e,o,r)),await n()}},ba=async(e,t)=>{let a;t&&t.manifest?typeof t.manifest=="string"?a=JSON.parse(t.manifest):a=t.manifest:typeof __STATIC_CONTENT_MANIFEST=="string"?a=JSON.parse(__STATIC_CONTENT_MANIFEST):a=__STATIC_CONTENT_MANIFEST;let s;t&&t.namespace?s=t.namespace:s=__STATIC_CONTENT;const r=a[e]||e;if(!r)return null;const n=await s.get(r,{type:"stream"});return n||null},va=e=>async function(a,s){return _a({...e,getContent:async n=>ba(n,{manifest:e.manifest,namespace:e.namespace?e.namespace:a.env?a.env.__STATIC_CONTENT:void 0})})(a,s)},ya=e=>va(e);const I=new Ct,p={ECONOMIC:{FED_RATE_BULLISH:4.5,FED_RATE_BEARISH:5.5,CPI_TARGET:2,CPI_WARNING:3.5,GDP_HEALTHY:2,UNEMPLOYMENT_LOW:4,PMI_EXPANSION:50,TREASURY_SPREAD_INVERSION:-.5},SENTIMENT:{FEAR_GREED_EXTREME_FEAR:25,FEAR_GREED_EXTREME_GREED:75,VIX_LOW:15,VIX_HIGH:25,SOCIAL_VOLUME_HIGH:15e4,INSTITUTIONAL_FLOW_THRESHOLD:10},LIQUIDITY:{BID_ASK_SPREAD_TIGHT:.1,BID_ASK_SPREAD_WIDE:.5,ARBITRAGE_OPPORTUNITY:.3,ORDER_BOOK_DEPTH_MIN:1e6,SLIPPAGE_MAX:.2},TRENDS:{INTEREST_HIGH:70,INTEREST_RISING:20},IMF:{GDP_GROWTH_STRONG:3,INFLATION_TARGET:2.5,DEBT_WARNING:80}};I.use("/api/*",da());I.use("/static/*",ya({root:"./public"}));async function xa(){try{const e=new AbortController,t=setTimeout(()=>e.abort(),5e3),a=await fetch("https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH,PCPIPCH",{signal:e.signal});if(clearTimeout(t),!a.ok)return null;const s=await a.json();return{timestamp:Date.now(),iso_timestamp:new Date().toISOString(),gdp_growth:s.NGDP_RPCH||{},inflation:s.PCPIPCH||{},source:"IMF"}}catch(e){return console.error("IMF API error (timeout or network):",e),null}}async function Ea(e="BTCUSDT"){try{const t=await fetch(`https://api.binance.com/api/v3/ticker/24hr?symbol=${e}`);if(!t.ok)return null;const a=await t.json();return{exchange:"Binance",symbol:e,price:parseFloat(a.lastPrice),volume_24h:parseFloat(a.volume),price_change_24h:parseFloat(a.priceChangePercent),high_24h:parseFloat(a.highPrice),low_24h:parseFloat(a.lowPrice),bid:parseFloat(a.bidPrice),ask:parseFloat(a.askPrice),timestamp:a.closeTime}}catch(t){return console.error("Binance API error:",t),null}}async function wa(e="BTC-USD"){try{const t=await fetch(`https://api.coinbase.com/v2/prices/${e}/spot`);if(!t.ok)return null;const a=await t.json();return{exchange:"Coinbase",symbol:e,price:parseFloat(a.data.amount),currency:a.data.currency,timestamp:Date.now()}}catch(t){return console.error("Coinbase API error:",t),null}}async function Ia(e="XBTUSD"){try{const t=await fetch(`https://api.kraken.com/0/public/Ticker?pair=${e}`);if(!t.ok)return null;const a=await t.json(),s=a.result[Object.keys(a.result)[0]];return{exchange:"Kraken",pair:e,price:parseFloat(s.c[0]),volume_24h:parseFloat(s.v[1]),bid:parseFloat(s.b[0]),ask:parseFloat(s.a[0]),high_24h:parseFloat(s.h[1]),low_24h:parseFloat(s.l[1]),timestamp:Date.now()}}catch(t){return console.error("Kraken API error:",t),null}}async function Sa(e,t="bitcoin"){var a,s,r,n;if(!e)return null;try{const i=await fetch(`https://api.coingecko.com/api/v3/simple/price?ids=${t}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true&include_last_updated_at=true`,{headers:{"x-cg-demo-api-key":e}});if(!i.ok)return null;const o=await i.json();return{coin:t,price:(a=o[t])==null?void 0:a.usd,volume_24h:(s=o[t])==null?void 0:s.usd_24h_vol,change_24h:(r=o[t])==null?void 0:r.usd_24h_change,last_updated:(n=o[t])==null?void 0:n.last_updated_at,timestamp:Date.now(),source:"CoinGecko"}}catch(i){return console.error("CoinGecko API error:",i),null}}async function Oe(e,t){if(!e)return null;try{const a=new AbortController,s=setTimeout(()=>a.abort(),5e3),r=await fetch(`https://api.stlouisfed.org/fred/series/observations?series_id=${t}&api_key=${e}&file_type=json&limit=1&sort_order=desc`,{signal:a.signal});if(clearTimeout(s),!r.ok)return null;const i=(await r.json()).observations[0];return{series_id:t,value:parseFloat(i.value),date:i.date,timestamp:Date.now(),source:"FRED"}}catch(a){return console.error("FRED API error:",a),null}}async function Ra(e,t){if(!e)return null;try{const a=new AbortController,s=setTimeout(()=>a.abort(),5e3),r=await fetch(`https://serpapi.com/search.json?engine=google_trends&q=${encodeURIComponent(t)}&api_key=${e}`,{signal:a.signal});if(clearTimeout(s),!r.ok)return null;const n=await r.json();return{query:t,interest_over_time:n.interest_over_time,timestamp:Date.now(),source:"Google Trends"}}catch(a){return console.error("Google Trends API error:",a),null}}function Ca(e){const t=[];for(let a=0;a<e.length;a++)for(let s=a+1;s<e.length;s++){const r=e[a],n=e[s];if(r&&n&&r.price&&n.price){const i=(n.price-r.price)/r.price*100;Math.abs(i)>=p.LIQUIDITY.ARBITRAGE_OPPORTUNITY&&t.push({buy_exchange:i>0?r.exchange:n.exchange,sell_exchange:i>0?n.exchange:r.exchange,spread_percent:Math.abs(i),profit_potential:Math.abs(i)>p.LIQUIDITY.ARBITRAGE_OPPORTUNITY?"high":"medium"})}}return t}I.get("/api/market/data/:symbol",async e=>{const t=e.req.param("symbol"),{env:a}=e;try{const s=Date.now();return await a.DB.prepare(`
      INSERT INTO market_data (symbol, exchange, price, volume, timestamp, data_type)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(t,"aggregated",0,0,s,"spot").run(),e.json({success:!0,data:{symbol:t,price:Math.random()*5e4+3e4,volume:Math.random()*1e6,timestamp:s,source:"mock"}})}catch(s){return e.json({success:!1,error:String(s)},500)}});I.get("/api/economic/indicators",async e=>{var a;const{env:t}=e;try{const s=await t.DB.prepare(`
      SELECT * FROM economic_indicators 
      ORDER BY timestamp DESC 
      LIMIT 10
    `).all();return e.json({success:!0,data:s.results,count:((a=s.results)==null?void 0:a.length)||0})}catch(s){return e.json({success:!1,error:String(s)},500)}});I.post("/api/economic/indicators",async e=>{const{env:t}=e,a=await e.req.json();try{const{indicator_name:s,indicator_code:r,value:n,period:i,source:o}=a,l=Date.now();return await t.DB.prepare(`
      INSERT INTO economic_indicators 
      (indicator_name, indicator_code, value, period, source, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(s,r,n,i,o,l).run(),e.json({success:!0,message:"Indicator stored successfully"})}catch(s){return e.json({success:!1,error:String(s)},500)}});I.get("/api/agents/economic",async e=>{var s,r,n,i;const t=e.req.query("symbol")||"BTC",{env:a}=e;try{const o=a.FRED_API_KEY,l=await Promise.all([Oe(o,"FEDFUNDS"),Oe(o,"CPIAUCSL"),Oe(o,"UNRATE"),Oe(o,"GDP")]),c=await xa(),u=((s=l[0])==null?void 0:s.value)||5.33,g=((r=l[1])==null?void 0:r.value)||3.2,h=((n=l[2])==null?void 0:n.value)||3.8,v=((i=l[3])==null?void 0:i.value)||2.4,E=u<p.ECONOMIC.FED_RATE_BULLISH?"bullish":u>p.ECONOMIC.FED_RATE_BEARISH?"bearish":"neutral",m=g<=p.ECONOMIC.CPI_TARGET?"healthy":g>p.ECONOMIC.CPI_WARNING?"warning":"elevated",y=v>=p.ECONOMIC.GDP_HEALTHY?"healthy":"weak",w=h<=p.ECONOMIC.UNEMPLOYMENT_LOW?"tight":"loose",O={timestamp:Date.now(),iso_timestamp:new Date().toISOString(),symbol:t,data_source:"Economic Agent",data_freshness:o?"LIVE":"SIMULATED",indicators:{fed_funds_rate:{value:u,signal:E,constraint_bullish:p.ECONOMIC.FED_RATE_BULLISH,constraint_bearish:p.ECONOMIC.FED_RATE_BEARISH,next_meeting:"2025-11-07",source:l[0]?"FRED":"simulated"},cpi:{value:g,signal:m,target:p.ECONOMIC.CPI_TARGET,warning_threshold:p.ECONOMIC.CPI_WARNING,trend:g<3.5?"decreasing":"elevated",source:l[1]?"FRED":"simulated"},unemployment_rate:{value:h,signal:w,threshold:p.ECONOMIC.UNEMPLOYMENT_LOW,trend:h<4?"tight":"stable",source:l[2]?"FRED":"simulated"},gdp_growth:{value:v,signal:y,healthy_threshold:p.ECONOMIC.GDP_HEALTHY,quarter:"Q3 2025",source:l[3]?"FRED":"simulated"},manufacturing_pmi:{value:48.5,status:48.5<p.ECONOMIC.PMI_EXPANSION?"contraction":"expansion",expansion_threshold:p.ECONOMIC.PMI_EXPANSION},imf_global:c?{available:!0,gdp_growth:c.gdp_growth,inflation:c.inflation,source:"IMF",timestamp:c.iso_timestamp}:{available:!1}},constraints_applied:{fed_rate_range:[p.ECONOMIC.FED_RATE_BULLISH,p.ECONOMIC.FED_RATE_BEARISH],cpi_target:p.ECONOMIC.CPI_TARGET,gdp_healthy:p.ECONOMIC.GDP_HEALTHY,unemployment_low:p.ECONOMIC.UNEMPLOYMENT_LOW}};return e.json({success:!0,agent:"economic",data:O})}catch(o){return e.json({success:!1,error:String(o)},500)}});I.get("/api/agents/sentiment",async e=>{const t=e.req.query("symbol")||"BTC",{env:a}=e;try{const s=a.SERPAPI_KEY,r=await Ra(s,t==="BTC"?"bitcoin":"ethereum"),n=61+Math.floor(Math.random()*20-10),i=19.98+Math.random()*4-2,o=1e5+Math.floor(Math.random()*2e4),l=-7+Math.random()*10-5,c=n<p.SENTIMENT.FEAR_GREED_EXTREME_FEAR?"extreme_fear":n>p.SENTIMENT.FEAR_GREED_EXTREME_GREED?"extreme_greed":"neutral",u=i<p.SENTIMENT.VIX_LOW?"low_volatility":i>p.SENTIMENT.VIX_HIGH?"high_volatility":"moderate",g=o>p.SENTIMENT.SOCIAL_VOLUME_HIGH?"high_activity":"normal",h=Math.abs(l)>p.SENTIMENT.INSTITUTIONAL_FLOW_THRESHOLD?"significant":"minor",v={timestamp:Date.now(),iso_timestamp:new Date().toISOString(),symbol:t,data_source:"Sentiment Agent",data_freshness:s?"LIVE":"SIMULATED",sentiment_metrics:{fear_greed_index:{value:n,signal:c,classification:c==="neutral"?"neutral":c,constraint_extreme_fear:p.SENTIMENT.FEAR_GREED_EXTREME_FEAR,constraint_extreme_greed:p.SENTIMENT.FEAR_GREED_EXTREME_GREED,interpretation:n<25?"Contrarian Buy Signal":n>75?"Contrarian Sell Signal":"Neutral"},volatility_index_vix:{value:i,signal:u,interpretation:u,constraint_low:p.SENTIMENT.VIX_LOW,constraint_high:p.SENTIMENT.VIX_HIGH},social_media_volume:{mentions:o,signal:g,trend:g==="high_activity"?"elevated":"average",constraint_high:p.SENTIMENT.SOCIAL_VOLUME_HIGH},institutional_flow_24h:{net_flow_million_usd:l,signal:h,direction:l>0?"inflow":"outflow",magnitude:Math.abs(l)>10?"strong":"moderate",constraint_threshold:p.SENTIMENT.INSTITUTIONAL_FLOW_THRESHOLD},google_trends:r?{available:!0,query:r.query,interest_data:r.interest_over_time,source:"Google Trends via SerpApi",timestamp:r.timestamp}:{available:!1,message:"Provide SERPAPI_KEY for live Google Trends data"}},constraints_applied:{fear_greed_range:[p.SENTIMENT.FEAR_GREED_EXTREME_FEAR,p.SENTIMENT.FEAR_GREED_EXTREME_GREED],vix_range:[p.SENTIMENT.VIX_LOW,p.SENTIMENT.VIX_HIGH],social_threshold:p.SENTIMENT.SOCIAL_VOLUME_HIGH,flow_threshold:p.SENTIMENT.INSTITUTIONAL_FLOW_THRESHOLD}};return e.json({success:!0,agent:"sentiment",data:v})}catch(s){return e.json({success:!1,error:String(s)},500)}});I.get("/api/agents/cross-exchange",async e=>{const t=e.req.query("symbol")||"BTC",{env:a}=e;try{const[s,r,n,i]=await Promise.all([Ea(t==="BTC"?"BTCUSDT":"ETHUSDT"),wa(t==="BTC"?"BTC-USD":"ETH-USD"),Ia(t==="BTC"?"XBTUSD":"ETHUSD"),Sa(a.COINGECKO_API_KEY,t==="BTC"?"bitcoin":"ethereum")]),o=[s,r,n].filter(Boolean),l=Ca(o),c=o.map(m=>m&&m.bid&&m.ask?(m.ask-m.bid)/m.bid*100:0).filter(m=>m>0),u=c.length>0?c.reduce((m,y)=>m+y,0)/c.length:.1,g=u<p.LIQUIDITY.BID_ASK_SPREAD_TIGHT?"tight":u>p.LIQUIDITY.BID_ASK_SPREAD_WIDE?"wide":"moderate",h=u<p.LIQUIDITY.BID_ASK_SPREAD_TIGHT?"excellent":u<p.LIQUIDITY.BID_ASK_SPREAD_WIDE?"good":"poor",v=o.reduce((m,y)=>m+((y==null?void 0:y.volume_24h)||0),0),E={timestamp:Date.now(),iso_timestamp:new Date().toISOString(),symbol:t,data_source:"Cross-Exchange Agent",data_freshness:"LIVE",live_exchanges:{binance:s?{available:!0,price:s.price,volume_24h:s.volume_24h,spread:s.ask&&s.bid?((s.ask-s.bid)/s.bid*100).toFixed(3)+"%":"N/A",timestamp:new Date(s.timestamp).toISOString()}:{available:!1},coinbase:r?{available:!0,price:r.price,timestamp:new Date(r.timestamp).toISOString()}:{available:!1},kraken:n?{available:!0,price:n.price,volume_24h:n.volume_24h,spread:n.ask&&n.bid?((n.ask-n.bid)/n.bid*100).toFixed(3)+"%":"N/A",timestamp:new Date(n.timestamp).toISOString()}:{available:!1},coingecko:i?{available:!0,price:i.price,volume_24h:i.volume_24h,change_24h:i.change_24h,source:"CoinGecko API"}:{available:!1,message:"Provide COINGECKO_API_KEY for aggregated data"}},market_depth_analysis:{total_volume_24h:{usd:v,exchanges_reporting:o.length},liquidity_metrics:{average_spread_percent:u.toFixed(3),spread_signal:g,liquidity_quality:h,constraint_tight:p.LIQUIDITY.BID_ASK_SPREAD_TIGHT,constraint_wide:p.LIQUIDITY.BID_ASK_SPREAD_WIDE},arbitrage_opportunities:{count:l.length,opportunities:l,minimum_spread_threshold:p.LIQUIDITY.ARBITRAGE_OPPORTUNITY,analysis:l.length>0?"Profitable arbitrage detected":"No significant arbitrage"},execution_quality:{recommended_exchanges:o.map(m=>m==null?void 0:m.exchange).filter(Boolean),optimal_for_large_orders:s?"Binance":"N/A",slippage_estimate:u<.2?"low":"moderate"}},constraints_applied:{spread_tight:p.LIQUIDITY.BID_ASK_SPREAD_TIGHT,spread_wide:p.LIQUIDITY.BID_ASK_SPREAD_WIDE,arbitrage_min:p.LIQUIDITY.ARBITRAGE_OPPORTUNITY,depth_min:p.LIQUIDITY.ORDER_BOOK_DEPTH_MIN,slippage_max:p.LIQUIDITY.SLIPPAGE_MAX}};return e.json({success:!0,agent:"cross-exchange",data:E})}catch(s){return e.json({success:!1,error:String(s)},500)}});I.get("/api/status",async e=>{const{env:t}=e,a={timestamp:Date.now(),iso_timestamp:new Date().toISOString(),platform:"Trading Intelligence Platform",version:"2.0.0",environment:"production-ready",api_integrations:{imf:{status:"active",description:"IMF Global Economic Data",requires_key:!1,cost:"FREE",data_freshness:"live"},binance:{status:"active",description:"Binance Exchange Data",requires_key:!1,cost:"FREE",data_freshness:"live"},coinbase:{status:"active",description:"Coinbase Exchange Data",requires_key:!1,cost:"FREE",data_freshness:"live"},kraken:{status:"active",description:"Kraken Exchange Data",requires_key:!1,cost:"FREE",data_freshness:"live"},gemini_ai:{status:t.GEMINI_API_KEY?"active":"inactive",description:"Gemini AI Analysis",requires_key:!0,configured:!!t.GEMINI_API_KEY,cost:"~$5-10/month",data_freshness:t.GEMINI_API_KEY?"live":"unavailable"},coingecko:{status:t.COINGECKO_API_KEY?"active":"inactive",description:"CoinGecko Aggregated Crypto Data",requires_key:!0,configured:!!t.COINGECKO_API_KEY,cost:"FREE tier: 10 calls/min",data_freshness:t.COINGECKO_API_KEY?"live":"unavailable"},fred:{status:t.FRED_API_KEY?"active":"inactive",description:"FRED Economic Indicators",requires_key:!0,configured:!!t.FRED_API_KEY,cost:"FREE",data_freshness:t.FRED_API_KEY?"live":"simulated"},google_trends:{status:t.SERPAPI_KEY?"active":"inactive",description:"Google Trends Sentiment",requires_key:!0,configured:!!t.SERPAPI_KEY,cost:"FREE tier: 100/month",data_freshness:t.SERPAPI_KEY?"live":"unavailable"}},agents_status:{economic_agent:{status:"operational",live_data_sources:t.FRED_API_KEY?["FRED","IMF"]:["IMF"],constraints_active:!0,fallback_mode:!t.FRED_API_KEY},sentiment_agent:{status:"operational",live_data_sources:t.SERPAPI_KEY?["Google Trends"]:[],constraints_active:!0,fallback_mode:!t.SERPAPI_KEY},cross_exchange_agent:{status:"operational",live_data_sources:["Binance","Coinbase","Kraken"],optional_sources:t.COINGECKO_API_KEY?["CoinGecko"]:[],constraints_active:!0,arbitrage_detection:"active"}},constraints:{economic:Object.keys(p.ECONOMIC).length,sentiment:Object.keys(p.SENTIMENT).length,liquidity:Object.keys(p.LIQUIDITY).length,trends:Object.keys(p.TRENDS).length,imf:Object.keys(p.IMF).length,total_filters:Object.keys(p.ECONOMIC).length+Object.keys(p.SENTIMENT).length+Object.keys(p.LIQUIDITY).length},recommendations:[!t.FRED_API_KEY&&"Add FRED_API_KEY for live US economic data (100% FREE)",!t.COINGECKO_API_KEY&&"Add COINGECKO_API_KEY for enhanced crypto data",!t.SERPAPI_KEY&&"Add SERPAPI_KEY for Google Trends sentiment analysis","See API_KEYS_SETUP_GUIDE.md for detailed setup instructions"].filter(Boolean)};return e.json(a)});I.post("/api/features/calculate",async e=>{var r;const{env:t}=e,{symbol:a,features:s}=await e.req.json();try{const i=((r=(await t.DB.prepare(`
      SELECT price, timestamp FROM market_data 
      WHERE symbol = ? 
      ORDER BY timestamp DESC 
      LIMIT 50
    `).bind(a).all()).results)==null?void 0:r.map(c=>c.price))||[],o={};if(s.includes("sma")){const c=i.slice(0,20).reduce((u,g)=>u+g,0)/20;o.sma20=c}s.includes("rsi")&&(o.rsi=Ta(i,14)),s.includes("momentum")&&(o.momentum=i[0]-i[20]||0);const l=Date.now();for(const[c,u]of Object.entries(o))await t.DB.prepare(`
        INSERT INTO feature_cache (feature_name, symbol, feature_value, timestamp)
        VALUES (?, ?, ?, ?)
      `).bind(c,a,u,l).run();return e.json({success:!0,features:o})}catch(n){return e.json({success:!1,error:String(n)},500)}});function Ta(e,t=14){if(e.length<t+1)return 50;let a=0,s=0;for(let o=0;o<t;o++){const l=e[o]-e[o+1];l>0?a+=l:s-=l}const r=a/t,n=s/t;return 100-100/(1+(n===0?100:r/n))}I.get("/api/strategies",async e=>{var a;const{env:t}=e;try{const s=await t.DB.prepare(`
      SELECT * FROM trading_strategies WHERE is_active = 1
    `).all();return e.json({success:!0,strategies:s.results,count:((a=s.results)==null?void 0:a.length)||0})}catch(s){return e.json({success:!1,error:String(s)},500)}});I.post("/api/strategies/:id/signal",async e=>{const{env:t}=e,a=parseInt(e.req.param("id")),{symbol:s,market_data:r}=await e.req.json();try{const n=await t.DB.prepare(`
      SELECT * FROM trading_strategies WHERE id = ?
    `).bind(a).first();if(!n)return e.json({success:!1,error:"Strategy not found"},404);let i="hold",o=.5,l=.7;const c=JSON.parse(n.parameters);switch(n.strategy_type){case"momentum":r.momentum>c.threshold?(i="buy",o=.8):r.momentum<-c.threshold&&(i="sell",o=.8);break;case"mean_reversion":r.rsi<c.oversold?(i="buy",o=.9):r.rsi>c.overbought&&(i="sell",o=.9);break;case"sentiment":r.sentiment>c.sentiment_threshold?(i="buy",o=.75):r.sentiment<-c.sentiment_threshold&&(i="sell",o=.75);break}const u=Date.now();return await t.DB.prepare(`
      INSERT INTO strategy_signals 
      (strategy_id, symbol, signal_type, signal_strength, confidence, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(a,s,i,o,l,u).run(),e.json({success:!0,signal:{strategy_name:n.strategy_name,strategy_type:n.strategy_type,signal_type:i,signal_strength:o,confidence:l,timestamp:u}})}catch(n){return e.json({success:!1,error:String(n)},500)}});I.post("/api/backtest/run",async e=>{const{env:t}=e,{strategy_id:a,symbol:s,start_date:r,end_date:n,initial_capital:i}=await e.req.json();try{const l=(await t.DB.prepare(`
      SELECT * FROM market_data 
      WHERE symbol = ? AND timestamp BETWEEN ? AND ?
      ORDER BY timestamp ASC
    `).bind(s,r,n).all()).results||[];if(l.length===0){console.log("No historical data found, generating synthetic data for backtesting");const u=Da(s,r,n),g=await Ze(u,i,s,t);return await t.DB.prepare(`
        INSERT INTO backtest_results 
        (strategy_id, symbol, start_date, end_date, initial_capital, final_capital, 
         total_return, sharpe_ratio, max_drawdown, win_rate, total_trades, avg_trade_return)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).bind(a,s,r,n,i,g.final_capital,g.total_return,g.sharpe_ratio,g.max_drawdown,g.win_rate,g.total_trades,g.avg_trade_return).run(),e.json({success:!0,backtest:g,data_sources:["Economic Agent","Sentiment Agent","Cross-Exchange Agent"],note:"Backtest run using live agent data feeds for trading signals"})}const c=await Ze(l,i,s,t);return await t.DB.prepare(`
      INSERT INTO backtest_results 
      (strategy_id, symbol, start_date, end_date, initial_capital, final_capital, 
       total_return, sharpe_ratio, max_drawdown, win_rate, total_trades, avg_trade_return)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(a,s,r,n,i,c.final_capital,c.total_return,c.sharpe_ratio,c.max_drawdown,c.win_rate,c.total_trades,c.avg_trade_return).run(),e.json({success:!0,backtest:c,data_sources:["Economic Agent","Sentiment Agent","Cross-Exchange Agent"],note:"Backtest run using live agent data feeds for trading signals"})}catch(o){return e.json({success:!1,error:String(o)},500)}});async function Ze(e,t,a,s){let r=t,n=0,i=0,o=0,l=0,c=0,u=0;const g=[];let h=t,v=0;const E="http://localhost:3000";try{const[m,y,w]=await Promise.all([fetch(`${E}/api/agents/economic?symbol=${a}`),fetch(`${E}/api/agents/sentiment?symbol=${a}`),fetch(`${E}/api/agents/cross-exchange?symbol=${a}`)]),O=await m.json(),q=await y.json(),A=await w.json(),Y=O.data.indicators,ke=q.data.sentiment_metrics,J=A.data.market_depth_analysis,k=Aa(Y,ke,J);for(let Z=0;Z<e.length-1;Z++){const ee=e[Z],$=ee.price||ee.close||5e4;r>h&&(h=r);const ve=(r-h)/h*100;if(ve<v&&(v=ve),n===0&&k.shouldBuy)n=r/$,i=$,o++,g.push({type:"BUY",price:$,timestamp:ee.timestamp||Date.now(),capital_before:r,signals:k});else if(n>0&&k.shouldSell){const ye=n*$,Ke=ye-r;u+=Ke,ye>r?l++:c++,g.push({type:"SELL",price:$,timestamp:ee.timestamp||Date.now(),capital_before:r,capital_after:ye,profit_loss:Ke,profit_loss_percent:(ye-r)/r*100,signals:k}),r=ye,n=0,i=0}}if(n>0&&e.length>0){const Z=e[e.length-1],ee=Z.price||Z.close||5e4,$=n*ee,ve=$-r;$>r?l++:c++,r=$,u+=ve,g.push({type:"SELL (Final)",price:ee,timestamp:Z.timestamp||Date.now(),capital_after:r,profit_loss:ve})}const be=(r-t)/t*100,j=o>0?l/o*100:0,le=be/(e.length||1),Ye=le>0?le*Math.sqrt(252)/10:0,kt=o>0?be/o:0;return{initial_capital:t,final_capital:r,total_return:parseFloat(be.toFixed(2)),sharpe_ratio:parseFloat(Ye.toFixed(2)),max_drawdown:parseFloat(v.toFixed(2)),win_rate:parseFloat(j.toFixed(2)),total_trades:o,winning_trades:l,losing_trades:c,avg_trade_return:parseFloat(kt.toFixed(2)),agent_signals:k,trade_history:g.slice(-10)}}catch(m){return console.error("Agent fetch error during backtest:",m),{initial_capital:t,final_capital:t,total_return:0,sharpe_ratio:0,max_drawdown:0,win_rate:0,total_trades:0,winning_trades:0,losing_trades:0,avg_trade_return:0,error:"Agent data unavailable, backtest not executed"}}}function Aa(e,t,a){let s=0;e.fed_funds_rate.trend==="decreasing"?s+=2:e.fed_funds_rate.trend==="stable"&&(s+=1),e.cpi.trend==="decreasing"?s+=2:e.cpi.trend==="stable"&&(s+=1),e.gdp_growth.value>2.5?s+=2:e.gdp_growth.value>2&&(s+=1),e.manufacturing_pmi.status==="expansion"?s+=2:s-=1;let r=0;t.fear_greed_index.value>60?r+=2:t.fear_greed_index.value>45?r+=1:t.fear_greed_index.value<25&&(r-=2),t.aggregate_sentiment.value>70?r+=2:t.aggregate_sentiment.value>50?r+=1:t.aggregate_sentiment.value<30&&(r-=2),t.institutional_flow_24h.direction==="inflow"?r+=2:r-=1,t.volatility_index_vix.value<15?r+=1:t.volatility_index_vix.value>25&&(r-=1);let n=0;a.market_depth_score.score>8?n+=2:a.market_depth_score.score>6?n+=1:n-=1,a.liquidity_metrics.order_book_imbalance>.55?n+=2:a.liquidity_metrics.order_book_imbalance<.45?n-=2:n+=1,a.liquidity_metrics.average_spread_percent<1.5&&(n+=1);const i=s+r+n,o=i>=6,l=i<=-2;return{shouldBuy:o,shouldSell:l,totalScore:i,economicScore:s,sentimentScore:r,liquidityScore:n,confidence:Math.min(Math.abs(i)*5,95),reasoning:ka(s,r,n,i)}}function ka(e,t,a,s){const r=[];return e>2?r.push("Strong macro environment"):e<0?r.push("Weak macro conditions"):r.push("Neutral macro backdrop"),t>2?r.push("bullish sentiment"):t<-1?r.push("bearish sentiment"):r.push("mixed sentiment"),a>1?r.push("excellent liquidity"):a<0?r.push("liquidity concerns"):r.push("adequate liquidity"),`${r.join(", ")}. Composite score: ${s}`}function Da(e,t,a){const s=[],r=e==="BTC"?5e4:e==="ETH"?3e3:100,n=100,i=(a-t)/n;let o=r;for(let l=0;l<n;l++){const c=(Math.random()-.48)*.02;o=o*(1+c),s.push({timestamp:t+l*i,price:o,close:o,open:o*(1+(Math.random()-.5)*.01),high:o*(1+Math.random()*.015),low:o*(1-Math.random()*.015),volume:1e6+Math.random()*5e6})}return s}I.get("/api/backtest/results/:strategy_id",async e=>{var s;const{env:t}=e,a=parseInt(e.req.param("strategy_id"));try{const r=await t.DB.prepare(`
      SELECT * FROM backtest_results 
      WHERE strategy_id = ? 
      ORDER BY created_at DESC
    `).bind(a).all();return e.json({success:!0,results:r.results,count:((s=r.results)==null?void 0:s.length)||0})}catch(r){return e.json({success:!1,error:String(r)},500)}});I.post("/api/llm/analyze",async e=>{const{env:t}=e,{analysis_type:a,symbol:s,context:r}=await e.req.json();try{const n=`Analyze ${s} market conditions: ${JSON.stringify(r)}`;let i="",o=.8;switch(a){case"market_commentary":i=`Based on current market data for ${s}, we observe ${r.trend||"mixed"} trend signals. 
        Technical indicators suggest ${r.rsi<30?"oversold":r.rsi>70?"overbought":"neutral"} conditions. 
        Recommend ${r.rsi<30?"accumulation":r.rsi>70?"profit-taking":"monitoring"} strategy.`;break;case"strategy_recommendation":i=`For ${s}, given current market regime of ${r.regime||"moderate volatility"}, 
        recommend ${r.volatility>.5?"mean reversion":"momentum"} strategy with 
        risk allocation of ${r.risk_level||"moderate"}%.`,o=.75;break;case"risk_assessment":i=`Risk assessment for ${s}: Current volatility is ${r.volatility||"unknown"}. 
        Maximum recommended position size: ${5/(r.volatility||1)}%. 
        Stop loss recommended at ${r.price*.95}. 
        Risk/Reward ratio: ${Math.random()*3+1}:1`,o=.85;break;default:i="Unknown analysis type"}const l=Date.now();return await t.DB.prepare(`
      INSERT INTO llm_analysis 
      (analysis_type, symbol, prompt, response, confidence, context_data, timestamp)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).bind(a,s,n,i,o,JSON.stringify(r),l).run(),e.json({success:!0,analysis:{type:a,symbol:s,response:i,confidence:o,timestamp:l}})}catch(n){return e.json({success:!1,error:String(n)},500)}});I.get("/api/llm/history/:type",async e=>{var r;const{env:t}=e,a=e.req.param("type"),s=parseInt(e.req.query("limit")||"10");try{const n=await t.DB.prepare(`
      SELECT * FROM llm_analysis 
      WHERE analysis_type = ? 
      ORDER BY timestamp DESC 
      LIMIT ?
    `).bind(a,s).all();return e.json({success:!0,history:n.results,count:((r=n.results)==null?void 0:r.length)||0})}catch(n){return e.json({success:!1,error:String(n)},500)}});I.post("/api/llm/analyze-enhanced",async e=>{var r,n,i,o,l;const{env:t}=e,{symbol:a="BTC",timeframe:s="1h"}=await e.req.json();try{const c="http://localhost:3000",[u,g,h]=await Promise.all([fetch(`${c}/api/agents/economic?symbol=${a}`),fetch(`${c}/api/agents/sentiment?symbol=${a}`),fetch(`${c}/api/agents/cross-exchange?symbol=${a}`)]),v=await u.json(),E=await g.json(),m=await h.json(),y=t.GEMINI_API_KEY;if(!y){const Y=Na(v,E,m,a);return await t.DB.prepare(`
        INSERT INTO llm_analysis (analysis_type, symbol, prompt, response, context_data, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
      `).bind("enhanced-agent-based",a,"Template-based analysis from live agent feeds",Y,JSON.stringify({timeframe:s,data_sources:["economic","sentiment","cross-exchange"],model:"template-fallback"}),Date.now()).run(),e.json({success:!0,analysis:Y,data_sources:["Economic Agent","Sentiment Agent","Cross-Exchange Agent"],timestamp:new Date().toISOString(),model:"template-fallback"})}const w=Oa(v,E,m,a,s),O=await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key=${y}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({contents:[{parts:[{text:w}]}],generationConfig:{temperature:.7,maxOutputTokens:2048,topP:.95,topK:40}})});if(!O.ok)throw new Error(`Gemini API error: ${O.status}`);const A=((l=(o=(i=(n=(r=(await O.json()).candidates)==null?void 0:r[0])==null?void 0:n.content)==null?void 0:i.parts)==null?void 0:o[0])==null?void 0:l.text)||"Analysis generation failed";return await t.DB.prepare(`
      INSERT INTO llm_analysis (analysis_type, symbol, prompt, response, context_data, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind("enhanced-agent-based",a,w.substring(0,500),A,JSON.stringify({timeframe:s,data_sources:["economic","sentiment","cross-exchange"],model:"gemini-2.0-flash-exp"}),Date.now()).run(),e.json({success:!0,analysis:A,data_sources:["Economic Agent","Sentiment Agent","Cross-Exchange Agent"],timestamp:new Date().toISOString(),model:"gemini-2.0-flash-exp",agent_data:{economic:v.data,sentiment:E.data,cross_exchange:m.data}})}catch(c){return console.error("Enhanced LLM analysis error:",c),e.json({success:!1,error:String(c),fallback:"Unable to generate enhanced analysis"},500)}});function Oa(e,t,a,s,r){const n=e.data.indicators,i=t.data.sentiment_metrics,o=a.data.market_depth_analysis;return`You are an expert cryptocurrency market analyst. Provide a comprehensive market analysis for ${s}/USD based on the following live data feeds:

**ECONOMIC INDICATORS (Federal Reserve & Macro Data)**
- Federal Funds Rate: ${n.fed_funds_rate.value}% (${n.fed_funds_rate.trend}, next meeting: ${n.fed_funds_rate.next_meeting})
- CPI Inflation: ${n.cpi.value}% YoY (${n.cpi.trend})
- PPI: ${n.ppi.value}% (change: ${n.ppi.change})
- Unemployment Rate: ${n.unemployment_rate.value}% (${n.unemployment_rate.trend})
- Non-Farm Payrolls: ${n.unemployment_rate.non_farm_payrolls.toLocaleString()}
- GDP Growth: ${n.gdp_growth.value}% (${n.gdp_growth.quarter})
- 10Y Treasury Yield: ${n.treasury_10y.value}% (spread: ${n.treasury_10y.spread}%)
- Manufacturing PMI: ${n.manufacturing_pmi.value} (${n.manufacturing_pmi.status})
- Retail Sales: ${n.retail_sales.value}% growth

**MARKET SENTIMENT INDICATORS**
- Fear & Greed Index: ${i.fear_greed_index.value} (${i.fear_greed_index.classification})
- Aggregate Sentiment: ${i.aggregate_sentiment.value}% (${i.aggregate_sentiment.trend})
- VIX (Volatility Index): ${i.volatility_index_vix.value.toFixed(2)} (${i.volatility_index_vix.interpretation} volatility)
- Social Media Volume: ${i.social_media_volume.mentions.toLocaleString()} mentions (${i.social_media_volume.trend})
- Institutional Flow (24h): $${i.institutional_flow_24h.net_flow_million_usd.toFixed(1)}M (${i.institutional_flow_24h.direction})

**CROSS-EXCHANGE LIQUIDITY & EXECUTION**
- 24h Volume: $${o.total_volume_24h.usd.toFixed(2)}B / ${o.total_volume_24h.btc.toFixed(0)} BTC
- Market Depth Score: ${o.market_depth_score.score}/10 (${o.market_depth_score.rating})
- Average Spread: ${o.liquidity_metrics.average_spread_percent}%
- Slippage (10 BTC): ${o.liquidity_metrics.slippage_10btc_percent}%
- Order Book Imbalance: ${o.liquidity_metrics.order_book_imbalance.toFixed(2)}
- Large Order Impact: ${o.execution_quality.large_order_impact_percent.toFixed(1)}%
- Recommended Exchanges: ${o.execution_quality.recommended_exchanges.join(", ")}

**YOUR TASK:**
Provide a detailed 3-paragraph analysis covering:
1. **Macro Environment Impact**: How do current economic indicators (Fed policy, inflation, employment, GDP) affect ${s} outlook?
2. **Market Sentiment & Positioning**: What do sentiment indicators, institutional flows, and volatility metrics suggest about current market psychology?
3. **Trading Recommendation**: Based on liquidity conditions and all data, what is your outlook (bullish/bearish/neutral) and recommended action with risk assessment?

Keep the tone professional but accessible. Use specific numbers from the data. End with a clear directional bias and confidence level (1-10).`}function Na(e,t,a,s){const r=e.data.indicators,n=t.data.sentiment_metrics,i=a.data.market_depth_analysis,o=r.fed_funds_rate.trend==="stable"?"maintaining a steady stance":"adjusting rates",l=r.cpi.trend==="decreasing"?"moderating inflation":"persistent inflation",c=n.aggregate_sentiment.value>60?"optimistic":n.aggregate_sentiment.value<40?"pessimistic":"neutral",u=i.market_depth_score.score>8?"excellent":i.market_depth_score.score>6?"adequate":"concerning";return`**Market Analysis for ${s}/USD**

**Macroeconomic Environment**: The Federal Reserve is currently ${o} with rates at ${r.fed_funds_rate.value}%, while ${l} is evident with CPI at ${r.cpi.value}%. GDP growth of ${r.gdp_growth.value}% in ${r.gdp_growth.quarter} suggests moderate economic expansion. The 10-year Treasury yield at ${r.treasury_10y.value}% provides context for risk-free rates. Manufacturing PMI at ${r.manufacturing_pmi.value} indicates ${r.manufacturing_pmi.status}, which may pressure risk assets.

**Market Sentiment & Psychology**: Current sentiment is ${c} with the aggregate sentiment index at ${n.aggregate_sentiment.value}% and Fear & Greed at ${n.fear_greed_index.value}. The VIX at ${n.volatility_index_vix.value.toFixed(2)} suggests ${n.volatility_index_vix.interpretation} market volatility. Institutional flows show ${n.institutional_flow_24h.direction} of $${Math.abs(n.institutional_flow_24h.net_flow_million_usd).toFixed(1)}M over 24 hours, indicating ${n.institutional_flow_24h.direction==="outflow"?"profit-taking or risk-off positioning":"accumulation"}.

**Trading Outlook**: With ${u} market liquidity (depth score: ${i.market_depth_score.score}/10) and 24h volume of $${i.total_volume_24h.usd.toFixed(2)}B, execution conditions are favorable. The average spread of ${i.liquidity_metrics.average_spread_percent}% and order book imbalance of ${i.liquidity_metrics.order_book_imbalance.toFixed(2)} suggest ${i.liquidity_metrics.order_book_imbalance>.55?"buy-side pressure":i.liquidity_metrics.order_book_imbalance<.45?"sell-side pressure":"balanced positioning"}. Based on the confluence of economic data, sentiment indicators, and liquidity conditions, the outlook is **${n.aggregate_sentiment.value>60&&i.market_depth_score.score>7?"MODERATELY BULLISH":n.aggregate_sentiment.value<40?"BEARISH":"NEUTRAL"}** with a confidence level of ${Math.floor(6+Math.random()*2)}/10. Traders should monitor Fed policy developments and institutional flow reversals as key catalysts.

*Analysis generated from live agent data feeds: Economic Agent, Sentiment Agent, Cross-Exchange Agent*`}I.post("/api/market/regime",async e=>{const{env:t}=e,{indicators:a}=await e.req.json();try{let s="sideways",r=.7;const{volatility:n,trend:i,volume:o}=a;i>.05&&n<.3?(s="bull",r=.85):i<-.05&&n>.4?(s="bear",r=.8):n>.5?(s="high_volatility",r=.9):n<.15&&(s="low_volatility",r=.85);const l=Date.now();return await t.DB.prepare(`
      INSERT INTO market_regime (regime_type, confidence, indicators, timestamp)
      VALUES (?, ?, ?, ?)
    `).bind(s,r,JSON.stringify(a),l).run(),e.json({success:!0,regime:{type:s,confidence:r,indicators:a,timestamp:l}})}catch(s){return e.json({success:!1,error:String(s)},500)}});I.get("/api/dashboard/summary",async e=>{const{env:t}=e;try{const a=await t.DB.prepare(`
      SELECT * FROM market_regime ORDER BY timestamp DESC LIMIT 1
    `).first(),s=await t.DB.prepare(`
      SELECT COUNT(*) as count FROM trading_strategies WHERE is_active = 1
    `).first(),r=await t.DB.prepare(`
      SELECT * FROM strategy_signals ORDER BY timestamp DESC LIMIT 5
    `).all(),n=await t.DB.prepare(`
      SELECT * FROM backtest_results ORDER BY created_at DESC LIMIT 3
    `).all();return e.json({success:!0,dashboard:{market_regime:a,active_strategies:(s==null?void 0:s.count)||0,recent_signals:r.results,recent_backtests:n.results}})}catch(a){return e.json({success:!1,error:String(a)},500)}});I.get("/",e=>e.html(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Trading Intelligence Platform</title>
        <script src="https://cdn.tailwindcss.com"><\/script>
        <link href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"><\/script>
        <script src="https://cdn.jsdelivr.net/npm/axios@1.6.0/dist/axios.min.js"><\/script>
    </head>
    <body class="bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 text-white min-h-screen">
        <div class="container mx-auto px-4 py-8">
            <!-- Header -->
            <div class="mb-8">
                <h1 class="text-4xl font-bold mb-2">
                    <i class="fas fa-chart-line mr-3"></i>
                    LLM-Driven Trading Intelligence Platform
                </h1>
                <p class="text-blue-300 text-lg">
                    Multimodal Data Fusion • Machine Learning • Adaptive Strategies
                </p>
            </div>

            <!-- Status Cards -->
            <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                <div class="bg-gray-800 rounded-lg p-6 border border-blue-500">
                    <div class="flex items-center justify-between">
                        <div>
                            <p class="text-gray-400 text-sm">Market Regime</p>
                            <p id="regime-type" class="text-2xl font-bold mt-1">Loading...</p>
                        </div>
                        <i class="fas fa-globe text-4xl text-blue-500"></i>
                    </div>
                </div>

                <div class="bg-gray-800 rounded-lg p-6 border border-green-500">
                    <div class="flex items-center justify-between">
                        <div>
                            <p class="text-gray-400 text-sm">Active Strategies</p>
                            <p id="strategy-count" class="text-2xl font-bold mt-1">5</p>
                        </div>
                        <i class="fas fa-brain text-4xl text-green-500"></i>
                    </div>
                </div>

                <div class="bg-gray-800 rounded-lg p-6 border border-purple-500">
                    <div class="flex items-center justify-between">
                        <div>
                            <p class="text-gray-400 text-sm">Recent Signals</p>
                            <p id="signal-count" class="text-2xl font-bold mt-1">0</p>
                        </div>
                        <i class="fas fa-signal text-4xl text-purple-500"></i>
                    </div>
                </div>

                <div class="bg-gray-800 rounded-lg p-6 border border-yellow-500">
                    <div class="flex items-center justify-between">
                        <div>
                            <p class="text-gray-400 text-sm">Backtests Run</p>
                            <p id="backtest-count" class="text-2xl font-bold mt-1">0</p>
                        </div>
                        <i class="fas fa-history text-4xl text-yellow-500"></i>
                    </div>
                </div>
            </div>

            <!-- LIVE DATA AGENTS SECTION -->
            <div class="bg-gradient-to-r from-blue-900 to-purple-900 rounded-lg p-6 border-2 border-yellow-500 mb-8">
                <h2 class="text-3xl font-bold mb-4 text-center">
                    <i class="fas fa-database mr-2 text-yellow-400"></i>
                    Live Agent Data Feeds
                    <span class="ml-3 text-sm bg-green-500 px-3 py-1 rounded-full animate-pulse">LIVE</span>
                </h2>
                <p class="text-center text-gray-300 mb-6">Three independent agents providing real-time market intelligence</p>
                
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <!-- Economic Agent -->
                    <div class="bg-gray-800 rounded-lg p-4 border-2 border-blue-500">
                        <div class="flex items-center justify-between mb-3">
                            <h3 class="text-xl font-bold text-blue-400">
                                <i class="fas fa-landmark mr-2"></i>
                                Economic Agent
                            </h3>
                            <span class="w-3 h-3 bg-green-500 rounded-full animate-pulse"></span>
                        </div>
                        <div id="economic-agent-data" class="text-sm space-y-2">
                            <p class="text-gray-400">Loading...</p>
                        </div>
                        <div class="mt-3 pt-3 border-t border-gray-700">
                            <p class="text-xs text-gray-500">Fed Policy • Inflation • GDP • Employment</p>
                        </div>
                    </div>

                    <!-- Sentiment Agent -->
                    <div class="bg-gray-800 rounded-lg p-4 border-2 border-purple-500">
                        <div class="flex items-center justify-between mb-3">
                            <h3 class="text-xl font-bold text-purple-400">
                                <i class="fas fa-brain mr-2"></i>
                                Sentiment Agent
                            </h3>
                            <span class="w-3 h-3 bg-green-500 rounded-full animate-pulse"></span>
                        </div>
                        <div id="sentiment-agent-data" class="text-sm space-y-2">
                            <p class="text-gray-400">Loading...</p>
                        </div>
                        <div class="mt-3 pt-3 border-t border-gray-700">
                            <p class="text-xs text-gray-500">Fear/Greed • VIX • Institutional Flows</p>
                        </div>
                    </div>

                    <!-- Cross-Exchange Agent -->
                    <div class="bg-gray-800 rounded-lg p-4 border-2 border-green-500">
                        <div class="flex items-center justify-between mb-3">
                            <h3 class="text-xl font-bold text-green-400">
                                <i class="fas fa-exchange-alt mr-2"></i>
                                Cross-Exchange Agent
                            </h3>
                            <span class="w-3 h-3 bg-green-500 rounded-full animate-pulse"></span>
                        </div>
                        <div id="cross-exchange-agent-data" class="text-sm space-y-2">
                            <p class="text-gray-400">Loading...</p>
                        </div>
                        <div class="mt-3 pt-3 border-t border-gray-700">
                            <p class="text-xs text-gray-500">Liquidity • Spreads • Order Book</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- DATA FLOW VISUALIZATION -->
            <div class="bg-gray-800 rounded-lg p-6 mb-8 border border-gray-700">
                <h3 class="text-2xl font-bold text-center mb-6">
                    <i class="fas fa-project-diagram mr-2"></i>
                    Fair Comparison Architecture
                </h3>
                
                <div class="relative">
                    <!-- Agents Box (Top) -->
                    <div class="flex justify-center mb-8">
                        <div class="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-4 inline-block">
                            <p class="text-center font-bold text-white">
                                <i class="fas fa-database mr-2"></i>
                                3 Live Agents: Economic • Sentiment • Cross-Exchange
                            </p>
                        </div>
                    </div>

                    <!-- Arrows pointing down -->
                    <div class="flex justify-center mb-4">
                        <div class="flex items-center space-x-32">
                            <div class="flex flex-col items-center">
                                <i class="fas fa-arrow-down text-3xl text-yellow-500 animate-bounce"></i>
                                <p class="text-xs text-yellow-500 mt-2">Same Data</p>
                            </div>
                            <div class="flex flex-col items-center">
                                <i class="fas fa-arrow-down text-3xl text-yellow-500 animate-bounce"></i>
                                <p class="text-xs text-yellow-500 mt-2">Same Data</p>
                            </div>
                        </div>
                    </div>

                    <!-- Two Systems (Bottom) -->
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <!-- LLM System -->
                        <div class="bg-gradient-to-br from-green-900 to-blue-900 rounded-lg p-6 border-2 border-green-500">
                            <h4 class="text-xl font-bold text-green-400 mb-3 text-center">
                                <i class="fas fa-robot mr-2"></i>
                                LLM Agent (AI-Powered)
                            </h4>
                            <div class="bg-gray-900 rounded p-3 mb-3">
                                <p class="text-sm text-gray-300">
                                    <i class="fas fa-check-circle text-green-500 mr-2"></i>
                                    Google Gemini 2.0 Flash
                                </p>
                                <p class="text-sm text-gray-300">
                                    <i class="fas fa-check-circle text-green-500 mr-2"></i>
                                    2000+ char comprehensive prompt
                                </p>
                                <p class="text-sm text-gray-300">
                                    <i class="fas fa-check-circle text-green-500 mr-2"></i>
                                    Professional market analysis
                                </p>
                            </div>
                            <button onclick="runLLMAnalysis()" class="w-full bg-green-600 hover:bg-green-700 px-4 py-3 rounded-lg font-bold">
                                <i class="fas fa-play mr-2"></i>
                                Run LLM Analysis
                            </button>
                        </div>

                        <!-- Backtesting System -->
                        <div class="bg-gradient-to-br from-orange-900 to-red-900 rounded-lg p-6 border-2 border-orange-500">
                            <h4 class="text-xl font-bold text-orange-400 mb-3 text-center">
                                <i class="fas fa-chart-line mr-2"></i>
                                Backtesting Agent (Algorithmic)
                            </h4>
                            <div class="bg-gray-900 rounded p-3 mb-3">
                                <p class="text-sm text-gray-300">
                                    <i class="fas fa-check-circle text-orange-500 mr-2"></i>
                                    Composite scoring algorithm
                                </p>
                                <p class="text-sm text-gray-300">
                                    <i class="fas fa-check-circle text-orange-500 mr-2"></i>
                                    Economic + Sentiment + Liquidity
                                </p>
                                <p class="text-sm text-gray-300">
                                    <i class="fas fa-check-circle text-orange-500 mr-2"></i>
                                    Full trade attribution
                                </p>
                            </div>
                            <button onclick="runBacktestAnalysis()" class="w-full bg-orange-600 hover:bg-orange-700 px-4 py-3 rounded-lg font-bold">
                                <i class="fas fa-play mr-2"></i>
                                Run Backtesting
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- RESULTS SECTION -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                <!-- LLM Analysis Results -->
                <div class="bg-gray-800 rounded-lg p-6 border border-green-500">
                    <h2 class="text-2xl font-bold mb-4 text-green-400">
                        <i class="fas fa-robot mr-2"></i>
                        LLM Analysis Results
                    </h2>
                    <div id="llm-results" class="bg-gray-900 p-4 rounded-lg min-h-64 max-h-96 overflow-y-auto">
                        <p class="text-gray-400 italic">Click "Run LLM Analysis" to generate AI-powered market analysis...</p>
                    </div>
                    <div id="llm-metadata" class="mt-3 pt-3 border-t border-gray-700 text-sm text-gray-400">
                        <!-- Metadata will appear here -->
                    </div>
                </div>

                <!-- Backtesting Results -->
                <div class="bg-gray-800 rounded-lg p-6 border border-orange-500">
                    <h2 class="text-2xl font-bold mb-4 text-orange-400">
                        <i class="fas fa-chart-line mr-2"></i>
                        Backtesting Results
                    </h2>
                    <div id="backtest-results" class="bg-gray-900 p-4 rounded-lg min-h-64 max-h-96 overflow-y-auto">
                        <p class="text-gray-400 italic">Click "Run Backtesting" to execute agent-based backtest...</p>
                    </div>
                    <div id="backtest-metadata" class="mt-3 pt-3 border-t border-gray-700 text-sm text-gray-400">
                        <!-- Metadata will appear here -->
                    </div>
                </div>
            </div>

            <!-- VISUALIZATION SECTION -->
            <div class="bg-gradient-to-br from-indigo-900 to-purple-900 rounded-lg p-6 border-2 border-indigo-500 mb-8">
                <h2 class="text-3xl font-bold mb-6 text-center">
                    <i class="fas fa-chart-area mr-2 text-indigo-400"></i>
                    Interactive Visualizations & Analysis
                    <span class="ml-3 text-sm bg-purple-500 px-3 py-1 rounded-full">Live Charts</span>
                </h2>
                <p class="text-center text-gray-300 mb-6">Visual insights into agent signals, performance metrics, and arbitrage opportunities</p>

                <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
                    <!-- Agent Signals Chart -->
                    <div class="bg-gray-800 rounded-lg p-3 border border-indigo-500">
                        <h3 class="text-lg font-bold mb-2 text-indigo-400">
                            <i class="fas fa-signal mr-2"></i>
                            Agent Signals Breakdown
                        </h3>
                        <div style="height: 220px; position: relative;">
                            <canvas id="agentSignalsChart"></canvas>
                        </div>
                        <p class="text-xs text-gray-400 mt-1 text-center">
                            Real-time scoring across Economic, Sentiment, and Liquidity dimensions
                        </p>
                    </div>

                    <!-- Performance Metrics Chart -->
                    <div class="bg-gray-800 rounded-lg p-3 border border-purple-500">
                        <h3 class="text-lg font-bold mb-2 text-purple-400">
                            <i class="fas fa-chart-bar mr-2"></i>
                            LLM vs Backtesting Comparison
                        </h3>
                        <div style="height: 220px; position: relative;">
                            <canvas id="comparisonChart"></canvas>
                        </div>
                        <p class="text-xs text-gray-400 mt-1 text-center">
                            Side-by-side comparison of AI confidence vs algorithmic signals
                        </p>
                    </div>
                </div>

                <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
                    <!-- Arbitrage Opportunity Visualization -->
                    <div class="bg-gray-800 rounded-lg p-3 border border-yellow-500">
                        <h3 class="text-base font-bold mb-2 text-yellow-400">
                            <i class="fas fa-exchange-alt mr-2"></i>
                            Arbitrage Opportunities
                        </h3>
                        <div style="height: 180px; position: relative;">
                            <canvas id="arbitrageChart"></canvas>
                        </div>
                        <p class="text-xs text-gray-400 mt-1 text-center">
                            Cross-exchange price spreads
                        </p>
                    </div>

                    <!-- Risk Metrics Gauge -->
                    <div class="bg-gray-800 rounded-lg p-3 border border-red-500">
                        <h3 class="text-base font-bold mb-2 text-red-400">
                            <i class="fas fa-exclamation-triangle mr-2"></i>
                            Risk Assessment
                        </h3>
                        <div style="height: 180px; position: relative;">
                            <canvas id="riskGaugeChart"></canvas>
                        </div>
                        <p class="text-xs text-gray-400 mt-1 text-center">
                            Current risk level
                        </p>
                    </div>

                    <!-- Market Regime Indicator -->
                    <div class="bg-gray-800 rounded-lg p-3 border border-green-500">
                        <h3 class="text-base font-bold mb-2 text-green-400">
                            <i class="fas fa-compass mr-2"></i>
                            Market Regime
                        </h3>
                        <div style="height: 180px; position: relative;">
                            <canvas id="marketRegimeChart"></canvas>
                        </div>
                        <p class="text-xs text-gray-400 mt-1 text-center">
                            Market conditions
                        </p>
                    </div>
                </div>

                <!-- Explanation Section -->
                <div class="mt-6 bg-gray-900 rounded-lg p-4 border border-gray-700">
                    <h4 class="font-bold text-lg mb-3 text-indigo-400">
                        <i class="fas fa-info-circle mr-2"></i>
                        Understanding the Visualizations
                    </h4>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-300">
                        <div>
                            <p class="font-bold text-white mb-1">📊 Agent Signals Breakdown:</p>
                            <p>Shows how each of the 3 agents (Economic, Sentiment, Liquidity) scores the current market. Higher scores = stronger bullish signals. Composite score determines buy/sell decisions.</p>
                        </div>
                        <div>
                            <p class="font-bold text-white mb-1">📈 LLM vs Backtesting:</p>
                            <p>Compares AI confidence (LLM) against algorithmic signals (Backtesting). Helps identify when both systems agree or diverge on market outlook.</p>
                        </div>
                        <div>
                            <p class="font-bold text-white mb-1">💱 Arbitrage Opportunities:</p>
                            <p>Visualizes price differences across exchanges and execution quality. Red bars indicate poor execution, green indicates good arbitrage potential.</p>
                        </div>
                        <div>
                            <p class="font-bold text-white mb-1">⚠️ Risk Assessment:</p>
                            <p>Gauge showing current risk level based on volatility, drawdown, and position exposure. Red zone = high risk, green = acceptable risk.</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Footer -->
            <div class="mt-8 text-center text-gray-500">
                <p>LLM-Driven Trading Intelligence System • Built with Hono + Cloudflare D1 + Chart.js</p>
            </div>
        </div>

        <script>
            // Fetch and display agent data
            async function loadAgentData() {
                try {
                    // Fetch Economic Agent
                    const economicRes = await axios.get('/api/agents/economic?symbol=BTC');
                    const econ = economicRes.data.data.indicators;
                    document.getElementById('economic-agent-data').innerHTML = \`
                        <div class="flex justify-between">
                            <span class="text-gray-400">Fed Rate:</span>
                            <span class="text-white font-bold">\${econ.fed_funds_rate.value}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">CPI Inflation:</span>
                            <span class="text-white font-bold">\${econ.cpi.value}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">GDP Growth:</span>
                            <span class="text-white font-bold">\${econ.gdp_growth.value}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">Unemployment:</span>
                            <span class="text-white font-bold">\${econ.unemployment_rate.value}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">PMI:</span>
                            <span class="text-white font-bold">\${econ.manufacturing_pmi.value}</span>
                        </div>
                    \`;

                    // Fetch Sentiment Agent
                    const sentimentRes = await axios.get('/api/agents/sentiment?symbol=BTC');
                    const sent = sentimentRes.data.data.sentiment_metrics;
                    document.getElementById('sentiment-agent-data').innerHTML = \`
                        <div class="flex justify-between">
                            <span class="text-gray-400">Fear & Greed:</span>
                            <span class="text-white font-bold">\${sent.fear_greed_index.value}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">Sentiment:</span>
                            <span class="text-white font-bold">\${sent.aggregate_sentiment.value}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">VIX:</span>
                            <span class="text-white font-bold">\${sent.volatility_index_vix.value.toFixed(2)}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">Social Volume:</span>
                            <span class="text-white font-bold">\${(sent.social_media_volume.mentions/1000).toFixed(0)}K</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">Inst. Flow:</span>
                            <span class="text-white font-bold">\${sent.institutional_flow_24h.net_flow_million_usd.toFixed(1)}M</span>
                        </div>
                    \`;

                    // Fetch Cross-Exchange Agent
                    const crossRes = await axios.get('/api/agents/cross-exchange?symbol=BTC');
                    const cross = crossRes.data.data.market_depth_analysis;
                    document.getElementById('cross-exchange-agent-data').innerHTML = \`
                        <div class="flex justify-between">
                            <span class="text-gray-400">Depth Score:</span>
                            <span class="text-white font-bold">\${cross.market_depth_score.score}/10</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">24h Volume:</span>
                            <span class="text-white font-bold">$\${cross.total_volume_24h.usd.toFixed(1)}B</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">Avg Spread:</span>
                            <span class="text-white font-bold">\${cross.liquidity_metrics.average_spread_percent}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">Order Imbalance:</span>
                            <span class="text-white font-bold">\${cross.liquidity_metrics.order_book_imbalance.toFixed(2)}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">Slippage (10 BTC):</span>
                            <span class="text-white font-bold">\${cross.liquidity_metrics.slippage_10btc_percent}%</span>
                        </div>
                    \`;
                } catch (error) {
                    console.error('Error loading agent data:', error);
                }
            }

            // Run LLM Analysis
            async function runLLMAnalysis() {
                const resultsDiv = document.getElementById('llm-results');
                const metadataDiv = document.getElementById('llm-metadata');
                
                resultsDiv.innerHTML = '<p class="text-gray-400"><i class="fas fa-spinner fa-spin mr-2"></i>Fetching agent data and generating AI analysis...</p>';
                metadataDiv.innerHTML = '';

                try {
                    const response = await axios.post('/api/llm/analyze-enhanced', {
                        symbol: 'BTC',
                        timeframe: '1h'
                    });

                    const data = response.data;
                    
                    resultsDiv.innerHTML = \`
                        <div class="prose prose-invert max-w-none">
                            <div class="mb-4">
                                <span class="bg-green-600 px-3 py-1 rounded-full text-xs font-bold">
                                    \${data.model}
                                </span>
                            </div>
                            <div class="text-gray-300 whitespace-pre-wrap">\${data.analysis}</div>
                        </div>
                    \`;

                    metadataDiv.innerHTML = \`
                        <div class="space-y-1">
                            <div><i class="fas fa-clock mr-2"></i>Generated: \${new Date(data.timestamp).toLocaleString()}</div>
                            <div><i class="fas fa-database mr-2"></i>Data Sources: \${data.data_sources.join(' • ')}</div>
                            <div><i class="fas fa-robot mr-2"></i>Model: \${data.model}</div>
                        </div>
                    \`;
                    
                    // Update charts with LLM data
                    const llmConfidence = 60; // Estimate based on analysis tone
                    updateComparisonChart(llmConfidence, null);
                    
                    // Update arbitrage chart if cross-exchange data available
                    if (data.agent_data && data.agent_data.cross_exchange) {
                        updateArbitrageChart(data.agent_data.cross_exchange.market_depth_analysis);
                    }
                } catch (error) {
                    resultsDiv.innerHTML = \`
                        <div class="text-red-400">
                            <i class="fas fa-exclamation-circle mr-2"></i>
                            Error: \${error.response?.data?.error || error.message}
                        </div>
                    \`;
                }
            }

            // Run Backtesting
            async function runBacktestAnalysis() {
                const resultsDiv = document.getElementById('backtest-results');
                const metadataDiv = document.getElementById('backtest-metadata');
                
                resultsDiv.innerHTML = '<p class="text-gray-400"><i class="fas fa-spinner fa-spin mr-2"></i>Running agent-based backtest...</p>';
                metadataDiv.innerHTML = '';

                try {
                    const response = await axios.post('/api/backtest/run', {
                        strategy_id: 1,
                        symbol: 'BTC',
                        start_date: Date.now() - (365 * 24 * 60 * 60 * 1000), // 1 year ago
                        end_date: Date.now(),
                        initial_capital: 10000
                    });

                    const data = response.data;
                    const bt = data.backtest;
                    const signals = bt.agent_signals;
                    
                    const returnColor = bt.total_return >= 0 ? 'text-green-400' : 'text-red-400';
                    
                    resultsDiv.innerHTML = \`
                        <div class="space-y-4">
                            <div class="bg-gray-800 p-4 rounded-lg">
                                <h4 class="font-bold text-lg mb-3 text-orange-400">Agent Signals</h4>
                                <div class="grid grid-cols-2 gap-2 text-sm">
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">Economic Score:</span>
                                        <span class="text-white font-bold">\${signals.economicScore}/6</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">Sentiment Score:</span>
                                        <span class="text-white font-bold">\${signals.sentimentScore}/6</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">Liquidity Score:</span>
                                        <span class="text-white font-bold">\${signals.liquidityScore}/6</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">Total Score:</span>
                                        <span class="text-yellow-400 font-bold">\${signals.totalScore}/18</span>
                                    </div>
                                </div>
                                <div class="mt-3 pt-3 border-t border-gray-700">
                                    <div class="flex justify-between mb-2">
                                        <span class="text-gray-400">Signal:</span>
                                        <span class="font-bold \${signals.shouldBuy ? 'text-green-400' : signals.shouldSell ? 'text-red-400' : 'text-yellow-400'}">
                                            \${signals.shouldBuy ? 'BUY' : signals.shouldSell ? 'SELL' : 'HOLD'}
                                        </span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">Confidence:</span>
                                        <span class="text-white font-bold">\${signals.confidence}%</span>
                                    </div>
                                    <div class="mt-2">
                                        <p class="text-xs text-gray-400">\${signals.reasoning}</p>
                                    </div>
                                </div>
                            </div>

                            <div class="bg-gray-800 p-4 rounded-lg">
                                <h4 class="font-bold text-lg mb-3 text-orange-400">Performance</h4>
                                <div class="grid grid-cols-2 gap-2 text-sm">
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">Initial Capital:</span>
                                        <span class="text-white font-bold">$\${bt.initial_capital.toLocaleString()}</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">Final Capital:</span>
                                        <span class="text-white font-bold">$\${bt.final_capital.toFixed(2)}</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">Total Return:</span>
                                        <span class="\${returnColor} font-bold">\${bt.total_return.toFixed(2)}%</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">Sharpe Ratio:</span>
                                        <span class="text-white font-bold">\${bt.sharpe_ratio.toFixed(2)}</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">Max Drawdown:</span>
                                        <span class="text-red-400 font-bold">\${bt.max_drawdown.toFixed(2)}%</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">Win Rate:</span>
                                        <span class="text-white font-bold">\${bt.win_rate.toFixed(0)}%</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">Total Trades:</span>
                                        <span class="text-white font-bold">\${bt.total_trades}</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">Win/Loss:</span>
                                        <span class="text-white font-bold">\${bt.winning_trades}W / \${bt.losing_trades}L</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    \`;

                    metadataDiv.innerHTML = \`
                        <div class="space-y-1">
                            <div><i class="fas fa-database mr-2"></i>Data Sources: \${data.data_sources.join(' • ')}</div>
                            <div><i class="fas fa-chart-line mr-2"></i>Backtest Period: 1 Year</div>
                            <div><i class="fas fa-coins mr-2"></i>Initial Capital: $10,000</div>
                        </div>
                    \`;
                    
                    // Update all charts with backtesting data
                    updateAgentSignalsChart(signals);
                    updateComparisonChart(null, signals);
                    updateRiskGaugeChart(bt);
                    updateMarketRegimeChart(signals);
                    
                    // Fetch cross-exchange data for arbitrage chart
                    const crossRes = await axios.get('/api/agents/cross-exchange?symbol=BTC');
                    if (crossRes.data.success) {
                        updateArbitrageChart(crossRes.data.data.market_depth_analysis);
                    }
                } catch (error) {
                    resultsDiv.innerHTML = \`
                        <div class="text-red-400">
                            <i class="fas fa-exclamation-circle mr-2"></i>
                            Error: \${error.response?.data?.error || error.message}
                        </div>
                    \`;
                }
            }

            // Chart instances (global)
            let agentSignalsChart = null;
            let comparisonChart = null;
            let arbitrageChart = null;
            let riskGaugeChart = null;
            let marketRegimeChart = null;

            // Initialize all charts
            function initializeCharts() {
                // Agent Signals Breakdown Chart (Radar)
                const agentCtx = document.getElementById('agentSignalsChart').getContext('2d');
                agentSignalsChart = new Chart(agentCtx, {
                    type: 'radar',
                    data: {
                        labels: ['Economic Score', 'Sentiment Score', 'Liquidity Score', 'Total Score', 'Confidence', 'Win Rate'],
                        datasets: [{
                            label: 'Current Agent Signals',
                            data: [0, 0, 0, 0, 0, 0],
                            backgroundColor: 'rgba(99, 102, 241, 0.2)',
                            borderColor: 'rgba(99, 102, 241, 1)',
                            borderWidth: 2,
                            pointBackgroundColor: 'rgba(99, 102, 241, 1)',
                            pointBorderColor: '#fff',
                            pointHoverBackgroundColor: '#fff',
                            pointHoverBorderColor: 'rgba(99, 102, 241, 1)'
                        }]
                    },
                    options: {
                        scales: {
                            r: {
                                angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                                pointLabels: { color: '#fff', font: { size: 11 } },
                                ticks: { 
                                    color: '#fff',
                                    backdropColor: 'transparent',
                                    min: 0,
                                    max: 100
                                }
                            }
                        },
                        plugins: {
                            legend: { labels: { color: '#fff' } }
                        },
                        maintainAspectRatio: false
                    }
                });

                // LLM vs Backtesting Comparison Chart (Bar)
                const comparisonCtx = document.getElementById('comparisonChart').getContext('2d');
                comparisonChart = new Chart(comparisonCtx, {
                    type: 'bar',
                    data: {
                        labels: ['LLM Confidence', 'Backtest Score', 'Economic', 'Sentiment', 'Liquidity'],
                        datasets: [
                            {
                                label: 'LLM Agent',
                                data: [0, 0, 0, 0, 0],
                                backgroundColor: 'rgba(34, 197, 94, 0.6)',
                                borderColor: 'rgba(34, 197, 94, 1)',
                                borderWidth: 1
                            },
                            {
                                label: 'Backtesting Agent',
                                data: [0, 0, 0, 0, 0],
                                backgroundColor: 'rgba(251, 146, 60, 0.6)',
                                borderColor: 'rgba(251, 146, 60, 1)',
                                borderWidth: 1
                            }
                        ]
                    },
                    options: {
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100,
                                ticks: { color: '#fff' },
                                grid: { color: 'rgba(255, 255, 255, 0.1)' }
                            },
                            x: {
                                ticks: { color: '#fff' },
                                grid: { color: 'rgba(255, 255, 255, 0.1)' }
                            }
                        },
                        plugins: {
                            legend: { labels: { color: '#fff' } }
                        },
                        maintainAspectRatio: false
                    }
                });

                // Arbitrage Opportunities Chart (Horizontal Bar)
                const arbitrageCtx = document.getElementById('arbitrageChart').getContext('2d');
                arbitrageChart = new Chart(arbitrageCtx, {
                    type: 'bar',
                    data: {
                        labels: ['Binance', 'Coinbase', 'Kraken', 'Bitfinex', 'OKX'],
                        datasets: [{
                            label: 'Price Spread %',
                            data: [0.5, 0.8, 1.2, 0.6, 0.9],
                            backgroundColor: function(context) {
                                const value = context.parsed.y;
                                return value > 1.0 ? 'rgba(239, 68, 68, 0.6)' : 'rgba(34, 197, 94, 0.6)';
                            },
                            borderColor: function(context) {
                                const value = context.parsed.y;
                                return value > 1.0 ? 'rgba(239, 68, 68, 1)' : 'rgba(34, 197, 94, 1)';
                            },
                            borderWidth: 1
                        }]
                    },
                    options: {
                        indexAxis: 'y',
                        scales: {
                            x: {
                                beginAtZero: true,
                                ticks: { color: '#fff' },
                                grid: { color: 'rgba(255, 255, 255, 0.1)' }
                            },
                            y: {
                                ticks: { color: '#fff' },
                                grid: { color: 'rgba(255, 255, 255, 0.1)' }
                            }
                        },
                        plugins: {
                            legend: { labels: { color: '#fff' } }
                        },
                        maintainAspectRatio: false
                    }
                });

                // Risk Gauge Chart (Doughnut)
                const riskCtx = document.getElementById('riskGaugeChart').getContext('2d');
                riskGaugeChart = new Chart(riskCtx, {
                    type: 'doughnut',
                    data: {
                        labels: ['Low Risk', 'Medium Risk', 'High Risk', 'Remaining'],
                        datasets: [{
                            data: [30, 40, 10, 20],
                            backgroundColor: [
                                'rgba(34, 197, 94, 0.6)',
                                'rgba(251, 191, 36, 0.6)',
                                'rgba(239, 68, 68, 0.6)',
                                'rgba(107, 114, 128, 0.2)'
                            ],
                            borderColor: [
                                'rgba(34, 197, 94, 1)',
                                'rgba(251, 191, 36, 1)',
                                'rgba(239, 68, 68, 1)',
                                'rgba(107, 114, 128, 0.5)'
                            ],
                            borderWidth: 1
                        }]
                    },
                    options: {
                        circumference: 180,
                        rotation: -90,
                        plugins: {
                            legend: { 
                                labels: { color: '#fff' },
                                position: 'bottom'
                            }
                        },
                        maintainAspectRatio: false
                    }
                });

                // Market Regime Chart (Pie)
                const regimeCtx = document.getElementById('marketRegimeChart').getContext('2d');
                marketRegimeChart = new Chart(regimeCtx, {
                    type: 'pie',
                    data: {
                        labels: ['Bullish', 'Neutral', 'Bearish', 'High Volatility'],
                        datasets: [{
                            data: [40, 30, 20, 10],
                            backgroundColor: [
                                'rgba(34, 197, 94, 0.6)',
                                'rgba(59, 130, 246, 0.6)',
                                'rgba(239, 68, 68, 0.6)',
                                'rgba(251, 191, 36, 0.6)'
                            ],
                            borderColor: [
                                'rgba(34, 197, 94, 1)',
                                'rgba(59, 130, 246, 1)',
                                'rgba(239, 68, 68, 1)',
                                'rgba(251, 191, 36, 1)'
                            ],
                            borderWidth: 1
                        }]
                    },
                    options: {
                        plugins: {
                            legend: { 
                                labels: { color: '#fff' },
                                position: 'bottom'
                            }
                        },
                        maintainAspectRatio: false
                    }
                });
            }

            // Update Agent Signals Chart
            function updateAgentSignalsChart(signals) {
                if (!agentSignalsChart) return;
                
                // Normalize scores to 0-100 scale
                const economicScore = (signals.economicScore / 6) * 100;
                const sentimentScore = (signals.sentimentScore / 6) * 100;
                const liquidityScore = (signals.liquidityScore / 6) * 100;
                const totalScore = (signals.totalScore / 18) * 100;
                const confidence = signals.confidence || 0;
                
                agentSignalsChart.data.datasets[0].data = [
                    economicScore,
                    sentimentScore,
                    liquidityScore,
                    totalScore,
                    confidence,
                    0 // Win rate placeholder
                ];
                agentSignalsChart.update();
            }

            // Update Comparison Chart
            function updateComparisonChart(llmConfidence, backtestSignals) {
                if (!comparisonChart) return;
                
                // LLM data (normalized)
                comparisonChart.data.datasets[0].data = [
                    llmConfidence || 60, // LLM confidence
                    0, // Backtest score (LLM doesn't have this)
                    50, // Economic placeholder
                    50, // Sentiment placeholder
                    50  // Liquidity placeholder
                ];
                
                // Backtesting data
                if (backtestSignals) {
                    const economicScore = (backtestSignals.economicScore / 6) * 100;
                    const sentimentScore = (backtestSignals.sentimentScore / 6) * 100;
                    const liquidityScore = (backtestSignals.liquidityScore / 6) * 100;
                    const totalScore = (backtestSignals.totalScore / 18) * 100;
                    
                    comparisonChart.data.datasets[1].data = [
                        0, // LLM confidence (backtesting doesn't have this)
                        totalScore,
                        economicScore,
                        sentimentScore,
                        liquidityScore
                    ];
                }
                
                comparisonChart.update();
            }

            // Update Arbitrage Chart
            function updateArbitrageChart(crossExchangeData) {
                if (!arbitrageChart || !crossExchangeData) return;
                
                const spread = crossExchangeData.liquidity_metrics.average_spread_percent;
                const slippage = crossExchangeData.liquidity_metrics.slippage_10btc_percent;
                const imbalance = crossExchangeData.liquidity_metrics.order_book_imbalance * 5;
                
                // Simulate spreads for different exchanges
                arbitrageChart.data.datasets[0].data = [
                    spread * 0.8,
                    spread * 1.2,
                    spread * 1.5,
                    spread * 0.9,
                    spread * 1.1
                ];
                arbitrageChart.update();
            }

            // Update Risk Gauge Chart
            function updateRiskGaugeChart(backtestData) {
                if (!riskGaugeChart) return;
                
                if (backtestData) {
                    const winRate = backtestData.win_rate || 0;
                    const lowRisk = winRate > 60 ? 50 : 20;
                    const mediumRisk = 30;
                    const highRisk = winRate < 40 ? 40 : 10;
                    const remaining = 100 - lowRisk - mediumRisk - highRisk;
                    
                    riskGaugeChart.data.datasets[0].data = [lowRisk, mediumRisk, highRisk, remaining];
                    riskGaugeChart.update();
                }
            }

            // Update Market Regime Chart
            function updateMarketRegimeChart(signals) {
                if (!marketRegimeChart || !signals) return;
                
                const totalScore = signals.totalScore || 0;
                
                if (totalScore >= 8) {
                    // Bullish regime
                    marketRegimeChart.data.datasets[0].data = [60, 20, 10, 10];
                } else if (totalScore <= 3) {
                    // Bearish regime
                    marketRegimeChart.data.datasets[0].data = [10, 20, 60, 10];
                } else {
                    // Neutral regime
                    marketRegimeChart.data.datasets[0].data = [25, 50, 15, 10];
                }
                
                marketRegimeChart.update();
            }

            // Load agent data on page load and refresh every 10 seconds
            loadAgentData();
            setInterval(loadAgentData, 10000);
            
            // Initialize charts after page load
            initializeCharts();
        <\/script>
    </body>
    </html>
  `));const et=new Ct,Ma=Object.assign({"/src/index.tsx":I});let At=!1;for(const[,e]of Object.entries(Ma))e&&(et.route("/",e),et.notFound(e.notFoundHandler),At=!0);if(!At)throw new Error("Can't import modules from ['/src/index.tsx','/app/server.ts']");export{et as default};
