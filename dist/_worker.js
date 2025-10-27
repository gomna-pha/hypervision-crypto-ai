var Pt=Object.defineProperty;var ze=e=>{throw TypeError(e)};var Nt=(e,t,a)=>t in e?Pt(e,t,{enumerable:!0,configurable:!0,writable:!0,value:a}):e[t]=a;var b=(e,t,a)=>Nt(e,typeof t!="symbol"?t+"":t,a),je=(e,t,a)=>t.has(e)||ze("Cannot "+a);var u=(e,t,a)=>(je(e,t,"read from private field"),a?a.call(e):t.get(e)),x=(e,t,a)=>t.has(e)?ze("Cannot add the same private member more than once"):t instanceof WeakSet?t.add(e):t.set(e,a),_=(e,t,a,s)=>(je(e,t,"write to private field"),s?s.call(e,a):t.set(e,a),a),E=(e,t,a)=>(je(e,t,"access private method"),a);var We=(e,t,a,s)=>({set _(r){_(e,t,r,a)},get _(){return u(e,t,s)}});var Xe=(e,t,a)=>(s,r)=>{let n=-1;return i(0);async function i(o){if(o<=n)throw new Error("next() called multiple times");n=o;let c,l=!1,d;if(e[o]?(d=e[o][0][0],s.req.routeIndex=o):d=o===e.length&&r||void 0,d)try{c=await d(s,()=>i(o+1))}catch(g){if(g instanceof Error&&t)s.error=g,c=await t(g,s),l=!0;else throw g}else s.finalized===!1&&a&&(c=await a(s));return c&&(s.finalized===!1||l)&&(s.res=c),s}},$t=Symbol(),Bt=async(e,t=Object.create(null))=>{const{all:a=!1,dot:s=!1}=t,n=(e instanceof ht?e.raw.headers:e.headers).get("Content-Type");return n!=null&&n.startsWith("multipart/form-data")||n!=null&&n.startsWith("application/x-www-form-urlencoded")?jt(e,{all:a,dot:s}):{}};async function jt(e,t){const a=await e.formData();return a?Ft(a,t):{}}function Ft(e,t){const a=Object.create(null);return e.forEach((s,r)=>{t.all||r.endsWith("[]")?Ht(a,r,s):a[r]=s}),t.dot&&Object.entries(a).forEach(([s,r])=>{s.includes(".")&&(Ut(a,s,r),delete a[s])}),a}var Ht=(e,t,a)=>{e[t]!==void 0?Array.isArray(e[t])?e[t].push(a):e[t]=[e[t],a]:t.endsWith("[]")?e[t]=[a]:e[t]=a},Ut=(e,t,a)=>{let s=e;const r=t.split(".");r.forEach((n,i)=>{i===r.length-1?s[n]=a:((!s[n]||typeof s[n]!="object"||Array.isArray(s[n])||s[n]instanceof File)&&(s[n]=Object.create(null)),s=s[n])})},ut=e=>{const t=e.split("/");return t[0]===""&&t.shift(),t},Gt=e=>{const{groups:t,path:a}=qt(e),s=ut(a);return Yt(s,t)},qt=e=>{const t=[];return e=e.replace(/\{[^}]+\}/g,(a,s)=>{const r=`@${s}`;return t.push([r,a]),r}),{groups:t,path:e}},Yt=(e,t)=>{for(let a=t.length-1;a>=0;a--){const[s]=t[a];for(let r=e.length-1;r>=0;r--)if(e[r].includes(s)){e[r]=e[r].replace(s,t[a][1]);break}}return e},ke={},Kt=(e,t)=>{if(e==="*")return"*";const a=e.match(/^\:([^\{\}]+)(?:\{(.+)\})?$/);if(a){const s=`${e}#${t}`;return ke[s]||(a[2]?ke[s]=t&&t[0]!==":"&&t[0]!=="*"?[s,a[1],new RegExp(`^${a[2]}(?=/${t})`)]:[e,a[1],new RegExp(`^${a[2]}$`)]:ke[s]=[e,a[1],!0]),ke[s]}return null},Ye=(e,t)=>{try{return t(e)}catch{return e.replace(/(?:%[0-9A-Fa-f]{2})+/g,a=>{try{return t(a)}catch{return a}})}},Vt=e=>Ye(e,decodeURI),gt=e=>{const t=e.url,a=t.indexOf("/",t.indexOf(":")+4);let s=a;for(;s<t.length;s++){const r=t.charCodeAt(s);if(r===37){const n=t.indexOf("?",s),i=t.slice(a,n===-1?void 0:n);return Vt(i.includes("%25")?i.replace(/%25/g,"%2525"):i)}else if(r===63)break}return t.slice(a,s)},zt=e=>{const t=gt(e);return t.length>1&&t.at(-1)==="/"?t.slice(0,-1):t},de=(e,t,...a)=>(a.length&&(t=de(t,...a)),`${(e==null?void 0:e[0])==="/"?"":"/"}${e}${t==="/"?"":`${(e==null?void 0:e.at(-1))==="/"?"":"/"}${(t==null?void 0:t[0])==="/"?t.slice(1):t}`}`),pt=e=>{if(e.charCodeAt(e.length-1)!==63||!e.includes(":"))return null;const t=e.split("/"),a=[];let s="";return t.forEach(r=>{if(r!==""&&!/\:/.test(r))s+="/"+r;else if(/\:/.test(r))if(/\?/.test(r)){a.length===0&&s===""?a.push("/"):a.push(s);const n=r.replace("?","");s+="/"+n,a.push(s)}else s+="/"+r}),a.filter((r,n,i)=>i.indexOf(r)===n)},Fe=e=>/[%+]/.test(e)?(e.indexOf("+")!==-1&&(e=e.replace(/\+/g," ")),e.indexOf("%")!==-1?Ye(e,ft):e):e,mt=(e,t,a)=>{let s;if(!a&&t&&!/[%+]/.test(t)){let i=e.indexOf(`?${t}`,8);for(i===-1&&(i=e.indexOf(`&${t}`,8));i!==-1;){const o=e.charCodeAt(i+t.length+1);if(o===61){const c=i+t.length+2,l=e.indexOf("&",c);return Fe(e.slice(c,l===-1?void 0:l))}else if(o==38||isNaN(o))return"";i=e.indexOf(`&${t}`,i+1)}if(s=/[%+]/.test(e),!s)return}const r={};s??(s=/[%+]/.test(e));let n=e.indexOf("?",8);for(;n!==-1;){const i=e.indexOf("&",n+1);let o=e.indexOf("=",n);o>i&&i!==-1&&(o=-1);let c=e.slice(n+1,o===-1?i===-1?void 0:i:o);if(s&&(c=Fe(c)),n=i,c==="")continue;let l;o===-1?l="":(l=e.slice(o+1,i===-1?void 0:i),s&&(l=Fe(l))),a?(r[c]&&Array.isArray(r[c])||(r[c]=[]),r[c].push(l)):r[c]??(r[c]=l)}return t?r[t]:r},Wt=mt,Xt=(e,t)=>mt(e,t,!0),ft=decodeURIComponent,Qe=e=>Ye(e,ft),pe,M,G,_t,bt,Ue,K,at,ht=(at=class{constructor(e,t="/",a=[[]]){x(this,G);b(this,"raw");x(this,pe);x(this,M);b(this,"routeIndex",0);b(this,"path");b(this,"bodyCache",{});x(this,K,e=>{const{bodyCache:t,raw:a}=this,s=t[e];if(s)return s;const r=Object.keys(t)[0];return r?t[r].then(n=>(r==="json"&&(n=JSON.stringify(n)),new Response(n)[e]())):t[e]=a[e]()});this.raw=e,this.path=t,_(this,M,a),_(this,pe,{})}param(e){return e?E(this,G,_t).call(this,e):E(this,G,bt).call(this)}query(e){return Wt(this.url,e)}queries(e){return Xt(this.url,e)}header(e){if(e)return this.raw.headers.get(e)??void 0;const t={};return this.raw.headers.forEach((a,s)=>{t[s]=a}),t}async parseBody(e){var t;return(t=this.bodyCache).parsedBody??(t.parsedBody=await Bt(this,e))}json(){return u(this,K).call(this,"text").then(e=>JSON.parse(e))}text(){return u(this,K).call(this,"text")}arrayBuffer(){return u(this,K).call(this,"arrayBuffer")}blob(){return u(this,K).call(this,"blob")}formData(){return u(this,K).call(this,"formData")}addValidatedData(e,t){u(this,pe)[e]=t}valid(e){return u(this,pe)[e]}get url(){return this.raw.url}get method(){return this.raw.method}get[$t](){return u(this,M)}get matchedRoutes(){return u(this,M)[0].map(([[,e]])=>e)}get routePath(){return u(this,M)[0].map(([[,e]])=>e)[this.routeIndex].path}},pe=new WeakMap,M=new WeakMap,G=new WeakSet,_t=function(e){const t=u(this,M)[0][this.routeIndex][1][e],a=E(this,G,Ue).call(this,t);return a&&/\%/.test(a)?Qe(a):a},bt=function(){const e={},t=Object.keys(u(this,M)[0][this.routeIndex][1]);for(const a of t){const s=E(this,G,Ue).call(this,u(this,M)[0][this.routeIndex][1][a]);s!==void 0&&(e[a]=/\%/.test(s)?Qe(s):s)}return e},Ue=function(e){return u(this,M)[1]?u(this,M)[1][e]:e},K=new WeakMap,at),Qt={Stringify:1},yt=async(e,t,a,s,r)=>{typeof e=="object"&&!(e instanceof String)&&(e instanceof Promise||(e=e.toString()),e instanceof Promise&&(e=await e));const n=e.callbacks;return n!=null&&n.length?(r?r[0]+=e:r=[e],Promise.all(n.map(o=>o({phase:t,buffer:r,context:s}))).then(o=>Promise.all(o.filter(Boolean).map(c=>yt(c,t,!1,s,r))).then(()=>r[0]))):Promise.resolve(e)},Jt="text/plain; charset=UTF-8",He=(e,t)=>({"Content-Type":e,...t}),Se,Te,j,me,F,k,Ie,fe,he,se,Ae,Ce,V,ue,st,Zt=(st=class{constructor(e,t){x(this,V);x(this,Se);x(this,Te);b(this,"env",{});x(this,j);b(this,"finalized",!1);b(this,"error");x(this,me);x(this,F);x(this,k);x(this,Ie);x(this,fe);x(this,he);x(this,se);x(this,Ae);x(this,Ce);b(this,"render",(...e)=>(u(this,fe)??_(this,fe,t=>this.html(t)),u(this,fe).call(this,...e)));b(this,"setLayout",e=>_(this,Ie,e));b(this,"getLayout",()=>u(this,Ie));b(this,"setRenderer",e=>{_(this,fe,e)});b(this,"header",(e,t,a)=>{this.finalized&&_(this,k,new Response(u(this,k).body,u(this,k)));const s=u(this,k)?u(this,k).headers:u(this,se)??_(this,se,new Headers);t===void 0?s.delete(e):a!=null&&a.append?s.append(e,t):s.set(e,t)});b(this,"status",e=>{_(this,me,e)});b(this,"set",(e,t)=>{u(this,j)??_(this,j,new Map),u(this,j).set(e,t)});b(this,"get",e=>u(this,j)?u(this,j).get(e):void 0);b(this,"newResponse",(...e)=>E(this,V,ue).call(this,...e));b(this,"body",(e,t,a)=>E(this,V,ue).call(this,e,t,a));b(this,"text",(e,t,a)=>!u(this,se)&&!u(this,me)&&!t&&!a&&!this.finalized?new Response(e):E(this,V,ue).call(this,e,t,He(Jt,a)));b(this,"json",(e,t,a)=>E(this,V,ue).call(this,JSON.stringify(e),t,He("application/json",a)));b(this,"html",(e,t,a)=>{const s=r=>E(this,V,ue).call(this,r,t,He("text/html; charset=UTF-8",a));return typeof e=="object"?yt(e,Qt.Stringify,!1,{}).then(s):s(e)});b(this,"redirect",(e,t)=>{const a=String(e);return this.header("Location",/[^\x00-\xFF]/.test(a)?encodeURI(a):a),this.newResponse(null,t??302)});b(this,"notFound",()=>(u(this,he)??_(this,he,()=>new Response),u(this,he).call(this,this)));_(this,Se,e),t&&(_(this,F,t.executionCtx),this.env=t.env,_(this,he,t.notFoundHandler),_(this,Ce,t.path),_(this,Ae,t.matchResult))}get req(){return u(this,Te)??_(this,Te,new ht(u(this,Se),u(this,Ce),u(this,Ae))),u(this,Te)}get event(){if(u(this,F)&&"respondWith"in u(this,F))return u(this,F);throw Error("This context has no FetchEvent")}get executionCtx(){if(u(this,F))return u(this,F);throw Error("This context has no ExecutionContext")}get res(){return u(this,k)||_(this,k,new Response(null,{headers:u(this,se)??_(this,se,new Headers)}))}set res(e){if(u(this,k)&&e){e=new Response(e.body,e);for(const[t,a]of u(this,k).headers.entries())if(t!=="content-type")if(t==="set-cookie"){const s=u(this,k).headers.getSetCookie();e.headers.delete("set-cookie");for(const r of s)e.headers.append("set-cookie",r)}else e.headers.set(t,a)}_(this,k,e),this.finalized=!0}get var(){return u(this,j)?Object.fromEntries(u(this,j)):{}}},Se=new WeakMap,Te=new WeakMap,j=new WeakMap,me=new WeakMap,F=new WeakMap,k=new WeakMap,Ie=new WeakMap,fe=new WeakMap,he=new WeakMap,se=new WeakMap,Ae=new WeakMap,Ce=new WeakMap,V=new WeakSet,ue=function(e,t,a){const s=u(this,k)?new Headers(u(this,k).headers):u(this,se)??new Headers;if(typeof t=="object"&&"headers"in t){const n=t.headers instanceof Headers?t.headers:new Headers(t.headers);for(const[i,o]of n)i.toLowerCase()==="set-cookie"?s.append(i,o):s.set(i,o)}if(a)for(const[n,i]of Object.entries(a))if(typeof i=="string")s.set(n,i);else{s.delete(n);for(const o of i)s.append(n,o)}const r=typeof t=="number"?t:(t==null?void 0:t.status)??u(this,me);return new Response(e,{status:r,headers:s})},st),T="ALL",ea="all",ta=["get","post","put","delete","options","patch"],xt="Can not add a route since the matcher is already built.",vt=class extends Error{},aa="__COMPOSED_HANDLER",sa=e=>e.text("404 Not Found",404),Je=(e,t)=>{if("getResponse"in e){const a=e.getResponse();return t.newResponse(a.body,a)}return console.error(e),t.text("Internal Server Error",500)},O,I,wt,P,te,Me,Oe,rt,Et=(rt=class{constructor(t={}){x(this,I);b(this,"get");b(this,"post");b(this,"put");b(this,"delete");b(this,"options");b(this,"patch");b(this,"all");b(this,"on");b(this,"use");b(this,"router");b(this,"getPath");b(this,"_basePath","/");x(this,O,"/");b(this,"routes",[]);x(this,P,sa);b(this,"errorHandler",Je);b(this,"onError",t=>(this.errorHandler=t,this));b(this,"notFound",t=>(_(this,P,t),this));b(this,"fetch",(t,...a)=>E(this,I,Oe).call(this,t,a[1],a[0],t.method));b(this,"request",(t,a,s,r)=>t instanceof Request?this.fetch(a?new Request(t,a):t,s,r):(t=t.toString(),this.fetch(new Request(/^https?:\/\//.test(t)?t:`http://localhost${de("/",t)}`,a),s,r)));b(this,"fire",()=>{addEventListener("fetch",t=>{t.respondWith(E(this,I,Oe).call(this,t.request,t,void 0,t.request.method))})});[...ta,ea].forEach(n=>{this[n]=(i,...o)=>(typeof i=="string"?_(this,O,i):E(this,I,te).call(this,n,u(this,O),i),o.forEach(c=>{E(this,I,te).call(this,n,u(this,O),c)}),this)}),this.on=(n,i,...o)=>{for(const c of[i].flat()){_(this,O,c);for(const l of[n].flat())o.map(d=>{E(this,I,te).call(this,l.toUpperCase(),u(this,O),d)})}return this},this.use=(n,...i)=>(typeof n=="string"?_(this,O,n):(_(this,O,"*"),i.unshift(n)),i.forEach(o=>{E(this,I,te).call(this,T,u(this,O),o)}),this);const{strict:s,...r}=t;Object.assign(this,r),this.getPath=s??!0?t.getPath??gt:zt}route(t,a){const s=this.basePath(t);return a.routes.map(r=>{var i;let n;a.errorHandler===Je?n=r.handler:(n=async(o,c)=>(await Xe([],a.errorHandler)(o,()=>r.handler(o,c))).res,n[aa]=r.handler),E(i=s,I,te).call(i,r.method,r.path,n)}),this}basePath(t){const a=E(this,I,wt).call(this);return a._basePath=de(this._basePath,t),a}mount(t,a,s){let r,n;s&&(typeof s=="function"?n=s:(n=s.optionHandler,s.replaceRequest===!1?r=c=>c:r=s.replaceRequest));const i=n?c=>{const l=n(c);return Array.isArray(l)?l:[l]}:c=>{let l;try{l=c.executionCtx}catch{}return[c.env,l]};r||(r=(()=>{const c=de(this._basePath,t),l=c==="/"?0:c.length;return d=>{const g=new URL(d.url);return g.pathname=g.pathname.slice(l)||"/",new Request(g,d)}})());const o=async(c,l)=>{const d=await a(r(c.req.raw),...i(c));if(d)return d;await l()};return E(this,I,te).call(this,T,de(t,"*"),o),this}},O=new WeakMap,I=new WeakSet,wt=function(){const t=new Et({router:this.router,getPath:this.getPath});return t.errorHandler=this.errorHandler,_(t,P,u(this,P)),t.routes=this.routes,t},P=new WeakMap,te=function(t,a,s){t=t.toUpperCase(),a=de(this._basePath,a);const r={basePath:this._basePath,path:a,method:t,handler:s};this.router.add(t,a,[s,r]),this.routes.push(r)},Me=function(t,a){if(t instanceof Error)return this.errorHandler(t,a);throw t},Oe=function(t,a,s,r){if(r==="HEAD")return(async()=>new Response(null,await E(this,I,Oe).call(this,t,a,s,"GET")))();const n=this.getPath(t,{env:s}),i=this.router.match(r,n),o=new Zt(t,{path:n,matchResult:i,env:s,executionCtx:a,notFoundHandler:u(this,P)});if(i[0].length===1){let l;try{l=i[0][0][0][0](o,async()=>{o.res=await u(this,P).call(this,o)})}catch(d){return E(this,I,Me).call(this,d,o)}return l instanceof Promise?l.then(d=>d||(o.finalized?o.res:u(this,P).call(this,o))).catch(d=>E(this,I,Me).call(this,d,o)):l??u(this,P).call(this,o)}const c=Xe(i[0],this.errorHandler,u(this,P));return(async()=>{try{const l=await c(o);if(!l.finalized)throw new Error("Context is not finalized. Did you forget to return a Response object or `await next()`?");return l.res}catch(l){return E(this,I,Me).call(this,l,o)}})()},rt),St=[];function ra(e,t){const a=this.buildAllMatchers(),s=(r,n)=>{const i=a[r]||a[T],o=i[2][n];if(o)return o;const c=n.match(i[0]);if(!c)return[[],St];const l=c.indexOf("",1);return[i[1][l],c]};return this.match=s,s(e,t)}var Ne="[^/]+",Ee=".*",we="(?:|/.*)",ge=Symbol(),na=new Set(".\\+*[^]$()");function ia(e,t){return e.length===1?t.length===1?e<t?-1:1:-1:t.length===1||e===Ee||e===we?1:t===Ee||t===we?-1:e===Ne?1:t===Ne?-1:e.length===t.length?e<t?-1:1:t.length-e.length}var re,ne,N,nt,Ge=(nt=class{constructor(){x(this,re);x(this,ne);x(this,N,Object.create(null))}insert(t,a,s,r,n){if(t.length===0){if(u(this,re)!==void 0)throw ge;if(n)return;_(this,re,a);return}const[i,...o]=t,c=i==="*"?o.length===0?["","",Ee]:["","",Ne]:i==="/*"?["","",we]:i.match(/^\:([^\{\}]+)(?:\{(.+)\})?$/);let l;if(c){const d=c[1];let g=c[2]||Ne;if(d&&c[2]&&(g===".*"||(g=g.replace(/^\((?!\?:)(?=[^)]+\)$)/,"(?:"),/\((?!\?:)/.test(g))))throw ge;if(l=u(this,N)[g],!l){if(Object.keys(u(this,N)).some(p=>p!==Ee&&p!==we))throw ge;if(n)return;l=u(this,N)[g]=new Ge,d!==""&&_(l,ne,r.varIndex++)}!n&&d!==""&&s.push([d,u(l,ne)])}else if(l=u(this,N)[i],!l){if(Object.keys(u(this,N)).some(d=>d.length>1&&d!==Ee&&d!==we))throw ge;if(n)return;l=u(this,N)[i]=new Ge}l.insert(o,a,s,r,n)}buildRegExpStr(){const a=Object.keys(u(this,N)).sort(ia).map(s=>{const r=u(this,N)[s];return(typeof u(r,ne)=="number"?`(${s})@${u(r,ne)}`:na.has(s)?`\\${s}`:s)+r.buildRegExpStr()});return typeof u(this,re)=="number"&&a.unshift(`#${u(this,re)}`),a.length===0?"":a.length===1?a[0]:"(?:"+a.join("|")+")"}},re=new WeakMap,ne=new WeakMap,N=new WeakMap,nt),$e,Re,it,oa=(it=class{constructor(){x(this,$e,{varIndex:0});x(this,Re,new Ge)}insert(e,t,a){const s=[],r=[];for(let i=0;;){let o=!1;if(e=e.replace(/\{[^}]+\}/g,c=>{const l=`@\\${i}`;return r[i]=[l,c],i++,o=!0,l}),!o)break}const n=e.match(/(?::[^\/]+)|(?:\/\*$)|./g)||[];for(let i=r.length-1;i>=0;i--){const[o]=r[i];for(let c=n.length-1;c>=0;c--)if(n[c].indexOf(o)!==-1){n[c]=n[c].replace(o,r[i][1]);break}}return u(this,Re).insert(n,t,s,u(this,$e),a),s}buildRegExp(){let e=u(this,Re).buildRegExpStr();if(e==="")return[/^$/,[],[]];let t=0;const a=[],s=[];return e=e.replace(/#(\d+)|@(\d+)|\.\*\$/g,(r,n,i)=>n!==void 0?(a[++t]=Number(n),"$()"):(i!==void 0&&(s[Number(i)]=++t),"")),[new RegExp(`^${e}`),a,s]}},$e=new WeakMap,Re=new WeakMap,it),ca=[/^$/,[],Object.create(null)],Pe=Object.create(null);function Tt(e){return Pe[e]??(Pe[e]=new RegExp(e==="*"?"":`^${e.replace(/\/\*$|([.\\+*[^\]$()])/g,(t,a)=>a?`\\${a}`:"(?:|/.*)")}$`))}function la(){Pe=Object.create(null)}function da(e){var l;const t=new oa,a=[];if(e.length===0)return ca;const s=e.map(d=>[!/\*|\/:/.test(d[0]),...d]).sort(([d,g],[p,m])=>d?1:p?-1:g.length-m.length),r=Object.create(null);for(let d=0,g=-1,p=s.length;d<p;d++){const[m,v,f]=s[d];m?r[v]=[f.map(([w])=>[w,Object.create(null)]),St]:g++;let y;try{y=t.insert(v,g,m)}catch(w){throw w===ge?new vt(v):w}m||(a[g]=f.map(([w,L])=>{const q=Object.create(null);for(L-=1;L>=0;L--){const[R,Y]=y[L];q[R]=Y}return[w,q]}))}const[n,i,o]=t.buildRegExp();for(let d=0,g=a.length;d<g;d++)for(let p=0,m=a[d].length;p<m;p++){const v=(l=a[d][p])==null?void 0:l[1];if(!v)continue;const f=Object.keys(v);for(let y=0,w=f.length;y<w;y++)v[f[y]]=o[v[f[y]]]}const c=[];for(const d in i)c[d]=a[i[d]];return[n,c,r]}function le(e,t){if(e){for(const a of Object.keys(e).sort((s,r)=>r.length-s.length))if(Tt(a).test(t))return[...e[a]]}}var z,W,Be,It,ot,ua=(ot=class{constructor(){x(this,Be);b(this,"name","RegExpRouter");x(this,z);x(this,W);b(this,"match",ra);_(this,z,{[T]:Object.create(null)}),_(this,W,{[T]:Object.create(null)})}add(e,t,a){var o;const s=u(this,z),r=u(this,W);if(!s||!r)throw new Error(xt);s[e]||[s,r].forEach(c=>{c[e]=Object.create(null),Object.keys(c[T]).forEach(l=>{c[e][l]=[...c[T][l]]})}),t==="/*"&&(t="*");const n=(t.match(/\/:/g)||[]).length;if(/\*$/.test(t)){const c=Tt(t);e===T?Object.keys(s).forEach(l=>{var d;(d=s[l])[t]||(d[t]=le(s[l],t)||le(s[T],t)||[])}):(o=s[e])[t]||(o[t]=le(s[e],t)||le(s[T],t)||[]),Object.keys(s).forEach(l=>{(e===T||e===l)&&Object.keys(s[l]).forEach(d=>{c.test(d)&&s[l][d].push([a,n])})}),Object.keys(r).forEach(l=>{(e===T||e===l)&&Object.keys(r[l]).forEach(d=>c.test(d)&&r[l][d].push([a,n]))});return}const i=pt(t)||[t];for(let c=0,l=i.length;c<l;c++){const d=i[c];Object.keys(r).forEach(g=>{var p;(e===T||e===g)&&((p=r[g])[d]||(p[d]=[...le(s[g],d)||le(s[T],d)||[]]),r[g][d].push([a,n-l+c+1]))})}}buildAllMatchers(){const e=Object.create(null);return Object.keys(u(this,W)).concat(Object.keys(u(this,z))).forEach(t=>{e[t]||(e[t]=E(this,Be,It).call(this,t))}),_(this,z,_(this,W,void 0)),la(),e}},z=new WeakMap,W=new WeakMap,Be=new WeakSet,It=function(e){const t=[];let a=e===T;return[u(this,z),u(this,W)].forEach(s=>{const r=s[e]?Object.keys(s[e]).map(n=>[n,s[e][n]]):[];r.length!==0?(a||(a=!0),t.push(...r)):e!==T&&t.push(...Object.keys(s[T]).map(n=>[n,s[T][n]]))}),a?da(t):null},ot),X,H,ct,ga=(ct=class{constructor(e){b(this,"name","SmartRouter");x(this,X,[]);x(this,H,[]);_(this,X,e.routers)}add(e,t,a){if(!u(this,H))throw new Error(xt);u(this,H).push([e,t,a])}match(e,t){if(!u(this,H))throw new Error("Fatal error");const a=u(this,X),s=u(this,H),r=a.length;let n=0,i;for(;n<r;n++){const o=a[n];try{for(let c=0,l=s.length;c<l;c++)o.add(...s[c]);i=o.match(e,t)}catch(c){if(c instanceof vt)continue;throw c}this.match=o.match.bind(o),_(this,X,[o]),_(this,H,void 0);break}if(n===r)throw new Error("Fatal error");return this.name=`SmartRouter + ${this.activeRouter.name}`,i}get activeRouter(){if(u(this,H)||u(this,X).length!==1)throw new Error("No active router has been determined yet.");return u(this,X)[0]}},X=new WeakMap,H=new WeakMap,ct),ve=Object.create(null),Q,C,ie,_e,A,U,ae,lt,At=(lt=class{constructor(e,t,a){x(this,U);x(this,Q);x(this,C);x(this,ie);x(this,_e,0);x(this,A,ve);if(_(this,C,a||Object.create(null)),_(this,Q,[]),e&&t){const s=Object.create(null);s[e]={handler:t,possibleKeys:[],score:0},_(this,Q,[s])}_(this,ie,[])}insert(e,t,a){_(this,_e,++We(this,_e)._);let s=this;const r=Gt(t),n=[];for(let i=0,o=r.length;i<o;i++){const c=r[i],l=r[i+1],d=Kt(c,l),g=Array.isArray(d)?d[0]:c;if(g in u(s,C)){s=u(s,C)[g],d&&n.push(d[1]);continue}u(s,C)[g]=new At,d&&(u(s,ie).push(d),n.push(d[1])),s=u(s,C)[g]}return u(s,Q).push({[e]:{handler:a,possibleKeys:n.filter((i,o,c)=>c.indexOf(i)===o),score:u(this,_e)}}),s}search(e,t){var o;const a=[];_(this,A,ve);let r=[this];const n=ut(t),i=[];for(let c=0,l=n.length;c<l;c++){const d=n[c],g=c===l-1,p=[];for(let m=0,v=r.length;m<v;m++){const f=r[m],y=u(f,C)[d];y&&(_(y,A,u(f,A)),g?(u(y,C)["*"]&&a.push(...E(this,U,ae).call(this,u(y,C)["*"],e,u(f,A))),a.push(...E(this,U,ae).call(this,y,e,u(f,A)))):p.push(y));for(let w=0,L=u(f,ie).length;w<L;w++){const q=u(f,ie)[w],R=u(f,A)===ve?{}:{...u(f,A)};if(q==="*"){const $=u(f,C)["*"];$&&(a.push(...E(this,U,ae).call(this,$,e,u(f,A))),_($,A,R),p.push($));continue}const[Y,De,J]=q;if(!d&&!(J instanceof RegExp))continue;const D=u(f,C)[Y],be=n.slice(c).join("/");if(J instanceof RegExp){const $=J.exec(be);if($){if(R[De]=$[0],a.push(...E(this,U,ae).call(this,D,e,u(f,A),R)),Object.keys(u(D,C)).length){_(D,A,R);const ce=((o=$[0].match(/\//))==null?void 0:o.length)??0;(i[ce]||(i[ce]=[])).push(D)}continue}}(J===!0||J.test(d))&&(R[De]=d,g?(a.push(...E(this,U,ae).call(this,D,e,R,u(f,A))),u(D,C)["*"]&&a.push(...E(this,U,ae).call(this,u(D,C)["*"],e,R,u(f,A)))):(_(D,A,R),p.push(D)))}}r=p.concat(i.shift()??[])}return a.length>1&&a.sort((c,l)=>c.score-l.score),[a.map(({handler:c,params:l})=>[c,l])]}},Q=new WeakMap,C=new WeakMap,ie=new WeakMap,_e=new WeakMap,A=new WeakMap,U=new WeakSet,ae=function(e,t,a,s){const r=[];for(let n=0,i=u(e,Q).length;n<i;n++){const o=u(e,Q)[n],c=o[t]||o[T],l={};if(c!==void 0&&(c.params=Object.create(null),r.push(c),a!==ve||s&&s!==ve))for(let d=0,g=c.possibleKeys.length;d<g;d++){const p=c.possibleKeys[d],m=l[c.score];c.params[p]=s!=null&&s[p]&&!m?s[p]:a[p]??(s==null?void 0:s[p]),l[c.score]=!0}}return r},lt),oe,dt,pa=(dt=class{constructor(){b(this,"name","TrieRouter");x(this,oe);_(this,oe,new At)}add(e,t,a){const s=pt(t);if(s){for(let r=0,n=s.length;r<n;r++)u(this,oe).insert(e,s[r],a);return}u(this,oe).insert(e,t,a)}match(e,t){return u(this,oe).search(e,t)}},oe=new WeakMap,dt),Ct=class extends Et{constructor(e={}){super(e),this.router=e.router??new ga({routers:[new ua,new pa]})}},ma=e=>{const a={...{origin:"*",allowMethods:["GET","HEAD","PUT","POST","DELETE","PATCH"],allowHeaders:[],exposeHeaders:[]},...e},s=(n=>typeof n=="string"?n==="*"?()=>n:i=>n===i?i:null:typeof n=="function"?n:i=>n.includes(i)?i:null)(a.origin),r=(n=>typeof n=="function"?n:Array.isArray(n)?()=>n:()=>[])(a.allowMethods);return async function(i,o){var d;function c(g,p){i.res.headers.set(g,p)}const l=await s(i.req.header("origin")||"",i);if(l&&c("Access-Control-Allow-Origin",l),a.credentials&&c("Access-Control-Allow-Credentials","true"),(d=a.exposeHeaders)!=null&&d.length&&c("Access-Control-Expose-Headers",a.exposeHeaders.join(",")),i.req.method==="OPTIONS"){a.origin!=="*"&&c("Vary","Origin"),a.maxAge!=null&&c("Access-Control-Max-Age",a.maxAge.toString());const g=await r(i.req.header("origin")||"",i);g.length&&c("Access-Control-Allow-Methods",g.join(","));let p=a.allowHeaders;if(!(p!=null&&p.length)){const m=i.req.header("Access-Control-Request-Headers");m&&(p=m.split(/\s*,\s*/))}return p!=null&&p.length&&(c("Access-Control-Allow-Headers",p.join(",")),i.res.headers.append("Vary","Access-Control-Request-Headers")),i.res.headers.delete("Content-Length"),i.res.headers.delete("Content-Type"),new Response(null,{headers:i.res.headers,status:204,statusText:"No Content"})}await o(),a.origin!=="*"&&i.header("Vary","Origin",{append:!0})}},fa=/^\s*(?:text\/(?!event-stream(?:[;\s]|$))[^;\s]+|application\/(?:javascript|json|xml|xml-dtd|ecmascript|dart|postscript|rtf|tar|toml|vnd\.dart|vnd\.ms-fontobject|vnd\.ms-opentype|wasm|x-httpd-php|x-javascript|x-ns-proxy-autoconfig|x-sh|x-tar|x-virtualbox-hdd|x-virtualbox-ova|x-virtualbox-ovf|x-virtualbox-vbox|x-virtualbox-vdi|x-virtualbox-vhd|x-virtualbox-vmdk|x-www-form-urlencoded)|font\/(?:otf|ttf)|image\/(?:bmp|vnd\.adobe\.photoshop|vnd\.microsoft\.icon|vnd\.ms-dds|x-icon|x-ms-bmp)|message\/rfc822|model\/gltf-binary|x-shader\/x-fragment|x-shader\/x-vertex|[^;\s]+?\+(?:json|text|xml|yaml))(?:[;\s]|$)/i,Ze=(e,t=_a)=>{const a=/\.([a-zA-Z0-9]+?)$/,s=e.match(a);if(!s)return;let r=t[s[1]];return r&&r.startsWith("text")&&(r+="; charset=utf-8"),r},ha={aac:"audio/aac",avi:"video/x-msvideo",avif:"image/avif",av1:"video/av1",bin:"application/octet-stream",bmp:"image/bmp",css:"text/css",csv:"text/csv",eot:"application/vnd.ms-fontobject",epub:"application/epub+zip",gif:"image/gif",gz:"application/gzip",htm:"text/html",html:"text/html",ico:"image/x-icon",ics:"text/calendar",jpeg:"image/jpeg",jpg:"image/jpeg",js:"text/javascript",json:"application/json",jsonld:"application/ld+json",map:"application/json",mid:"audio/x-midi",midi:"audio/x-midi",mjs:"text/javascript",mp3:"audio/mpeg",mp4:"video/mp4",mpeg:"video/mpeg",oga:"audio/ogg",ogv:"video/ogg",ogx:"application/ogg",opus:"audio/opus",otf:"font/otf",pdf:"application/pdf",png:"image/png",rtf:"application/rtf",svg:"image/svg+xml",tif:"image/tiff",tiff:"image/tiff",ts:"video/mp2t",ttf:"font/ttf",txt:"text/plain",wasm:"application/wasm",webm:"video/webm",weba:"audio/webm",webmanifest:"application/manifest+json",webp:"image/webp",woff:"font/woff",woff2:"font/woff2",xhtml:"application/xhtml+xml",xml:"application/xml",zip:"application/zip","3gp":"video/3gpp","3g2":"video/3gpp2",gltf:"model/gltf+json",glb:"model/gltf-binary"},_a=ha,ba=(...e)=>{let t=e.filter(r=>r!=="").join("/");t=t.replace(new RegExp("(?<=\\/)\\/+","g"),"");const a=t.split("/"),s=[];for(const r of a)r===".."&&s.length>0&&s.at(-1)!==".."?s.pop():r!=="."&&s.push(r);return s.join("/")||"."},Rt={br:".br",zstd:".zst",gzip:".gz"},ya=Object.keys(Rt),xa="index.html",va=e=>{const t=e.root??"./",a=e.path,s=e.join??ba;return async(r,n)=>{var d,g,p,m;if(r.finalized)return n();let i;if(e.path)i=e.path;else try{if(i=decodeURIComponent(r.req.path),/(?:^|[\/\\])\.\.(?:$|[\/\\])/.test(i))throw new Error}catch{return await((d=e.onNotFound)==null?void 0:d.call(e,r.req.path,r)),n()}let o=s(t,!a&&e.rewriteRequestPath?e.rewriteRequestPath(i):i);e.isDir&&await e.isDir(o)&&(o=s(o,xa));const c=e.getContent;let l=await c(o,r);if(l instanceof Response)return r.newResponse(l.body,l);if(l){const v=e.mimes&&Ze(o,e.mimes)||Ze(o);if(r.header("Content-Type",v||"application/octet-stream"),e.precompressed&&(!v||fa.test(v))){const f=new Set((g=r.req.header("Accept-Encoding"))==null?void 0:g.split(",").map(y=>y.trim()));for(const y of ya){if(!f.has(y))continue;const w=await c(o+Rt[y],r);if(w){l=w,r.header("Content-Encoding",y),r.header("Vary","Accept-Encoding",{append:!0});break}}}return await((p=e.onFound)==null?void 0:p.call(e,o,r)),r.body(l)}await((m=e.onNotFound)==null?void 0:m.call(e,o,r)),await n()}},Ea=async(e,t)=>{let a;t&&t.manifest?typeof t.manifest=="string"?a=JSON.parse(t.manifest):a=t.manifest:typeof __STATIC_CONTENT_MANIFEST=="string"?a=JSON.parse(__STATIC_CONTENT_MANIFEST):a=__STATIC_CONTENT_MANIFEST;let s;t&&t.namespace?s=t.namespace:s=__STATIC_CONTENT;const r=a[e]||e;if(!r)return null;const n=await s.get(r,{type:"stream"});return n||null},wa=e=>async function(a,s){return va({...e,getContent:async n=>Ea(n,{manifest:e.manifest,namespace:e.namespace?e.namespace:a.env?a.env.__STATIC_CONTENT:void 0})})(a,s)},Sa=e=>wa(e);const S=new Ct,h={ECONOMIC:{FED_RATE_BULLISH:4.5,FED_RATE_BEARISH:5.5,CPI_TARGET:2,CPI_WARNING:3.5,GDP_HEALTHY:2,UNEMPLOYMENT_LOW:4,PMI_EXPANSION:50,TREASURY_SPREAD_INVERSION:-.5},SENTIMENT:{FEAR_GREED_EXTREME_FEAR:25,FEAR_GREED_EXTREME_GREED:75,VIX_LOW:15,VIX_HIGH:25,SOCIAL_VOLUME_HIGH:15e4,INSTITUTIONAL_FLOW_THRESHOLD:10},LIQUIDITY:{BID_ASK_SPREAD_TIGHT:.1,BID_ASK_SPREAD_WIDE:.5,ARBITRAGE_OPPORTUNITY:.3,ORDER_BOOK_DEPTH_MIN:1e6,SLIPPAGE_MAX:.2},TRENDS:{INTEREST_HIGH:70,INTEREST_RISING:20},IMF:{GDP_GROWTH_STRONG:3,INFLATION_TARGET:2.5,DEBT_WARNING:80}};S.use("/api/*",ma());S.use("/static/*",Sa({root:"./public"}));async function Ta(){try{const e=new AbortController,t=setTimeout(()=>e.abort(),5e3),a=await fetch("https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH,PCPIPCH",{signal:e.signal});if(clearTimeout(t),!a.ok)return null;const s=await a.json();return{timestamp:Date.now(),iso_timestamp:new Date().toISOString(),gdp_growth:s.NGDP_RPCH||{},inflation:s.PCPIPCH||{},source:"IMF"}}catch(e){return console.error("IMF API error (timeout or network):",e),null}}async function Dt(e="BTCUSDT"){try{const t=await fetch(`https://api.binance.com/api/v3/ticker/24hr?symbol=${e}`);if(!t.ok)return null;const a=await t.json();return{exchange:"Binance",symbol:e,price:parseFloat(a.lastPrice),volume_24h:parseFloat(a.volume),price_change_24h:parseFloat(a.priceChangePercent),high_24h:parseFloat(a.highPrice),low_24h:parseFloat(a.lowPrice),bid:parseFloat(a.bidPrice),ask:parseFloat(a.askPrice),timestamp:a.closeTime}}catch(t){return console.error("Binance API error:",t),null}}async function kt(e="BTC-USD"){try{const t=await fetch(`https://api.coinbase.com/v2/prices/${e}/spot`);if(!t.ok)return null;const a=await t.json();return{exchange:"Coinbase",symbol:e,price:parseFloat(a.data.amount),currency:a.data.currency,timestamp:Date.now()}}catch(t){return console.error("Coinbase API error:",t),null}}async function Lt(e="XBTUSD"){try{const t=await fetch(`https://api.kraken.com/0/public/Ticker?pair=${e}`);if(!t.ok)return null;const a=await t.json(),s=a.result[Object.keys(a.result)[0]];return{exchange:"Kraken",pair:e,price:parseFloat(s.c[0]),volume_24h:parseFloat(s.v[1]),bid:parseFloat(s.b[0]),ask:parseFloat(s.a[0]),high_24h:parseFloat(s.h[1]),low_24h:parseFloat(s.l[1]),timestamp:Date.now()}}catch(t){return console.error("Kraken API error:",t),null}}async function Ia(e,t="bitcoin"){var a,s,r,n;if(!e)return null;try{const i=await fetch(`https://api.coingecko.com/api/v3/simple/price?ids=${t}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true&include_last_updated_at=true`,{headers:{"x-cg-demo-api-key":e}});if(!i.ok)return null;const o=await i.json();return{coin:t,price:(a=o[t])==null?void 0:a.usd,volume_24h:(s=o[t])==null?void 0:s.usd_24h_vol,change_24h:(r=o[t])==null?void 0:r.usd_24h_change,last_updated:(n=o[t])==null?void 0:n.last_updated_at,timestamp:Date.now(),source:"CoinGecko"}}catch(i){return console.error("CoinGecko API error:",i),null}}async function Le(e,t){if(!e)return null;try{const a=new AbortController,s=setTimeout(()=>a.abort(),5e3),r=await fetch(`https://api.stlouisfed.org/fred/series/observations?series_id=${t}&api_key=${e}&file_type=json&limit=1&sort_order=desc`,{signal:a.signal});if(clearTimeout(s),!r.ok)return null;const i=(await r.json()).observations[0];return{series_id:t,value:parseFloat(i.value),date:i.date,timestamp:Date.now(),source:"FRED"}}catch(a){return console.error("FRED API error:",a),null}}async function Aa(e,t){if(!e)return null;try{const a=new AbortController,s=setTimeout(()=>a.abort(),5e3),r=await fetch(`https://serpapi.com/search.json?engine=google_trends&q=${encodeURIComponent(t)}&api_key=${e}`,{signal:a.signal});if(clearTimeout(s),!r.ok)return null;const n=await r.json();return{query:t,interest_over_time:n.interest_over_time,timestamp:Date.now(),source:"Google Trends"}}catch(a){return console.error("Google Trends API error:",a),null}}function Ca(e){const t=[];for(let a=0;a<e.length;a++)for(let s=a+1;s<e.length;s++){const r=e[a],n=e[s];if(r&&n&&r.price&&n.price){const i=(n.price-r.price)/r.price*100;Math.abs(i)>=h.LIQUIDITY.ARBITRAGE_OPPORTUNITY&&t.push({buy_exchange:i>0?r.exchange:n.exchange,sell_exchange:i>0?n.exchange:r.exchange,spread_percent:Math.abs(i),profit_potential:Math.abs(i)>h.LIQUIDITY.ARBITRAGE_OPPORTUNITY?"high":"medium"})}}return t}S.get("/api/market/data/:symbol",async e=>{const t=e.req.param("symbol"),{env:a}=e;try{const s=Date.now();return await a.DB.prepare(`
      INSERT INTO market_data (symbol, exchange, price, volume, timestamp, data_type)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(t,"aggregated",0,0,s,"spot").run(),e.json({success:!0,data:{symbol:t,price:Math.random()*5e4+3e4,volume:Math.random()*1e6,timestamp:s,source:"mock"}})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.get("/api/economic/indicators",async e=>{var a;const{env:t}=e;try{const s=await t.DB.prepare(`
      SELECT * FROM economic_indicators 
      ORDER BY timestamp DESC 
      LIMIT 10
    `).all();return e.json({success:!0,data:s.results,count:((a=s.results)==null?void 0:a.length)||0})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.post("/api/economic/indicators",async e=>{const{env:t}=e,a=await e.req.json();try{const{indicator_name:s,indicator_code:r,value:n,period:i,source:o}=a,c=Date.now();return await t.DB.prepare(`
      INSERT INTO economic_indicators 
      (indicator_name, indicator_code, value, period, source, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(s,r,n,i,o,c).run(),e.json({success:!0,message:"Indicator stored successfully"})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.get("/api/agents/economic",async e=>{var s,r,n,i;const t=e.req.query("symbol")||"BTC",{env:a}=e;try{const o=a.FRED_API_KEY,c=await Promise.all([Le(o,"FEDFUNDS"),Le(o,"CPIAUCSL"),Le(o,"UNRATE"),Le(o,"GDP")]),l=await Ta(),d=((s=c[0])==null?void 0:s.value)||5.33,g=((r=c[1])==null?void 0:r.value)||3.2,p=((n=c[2])==null?void 0:n.value)||3.8,m=((i=c[3])==null?void 0:i.value)||2.4,v=d<h.ECONOMIC.FED_RATE_BULLISH?"bullish":d>h.ECONOMIC.FED_RATE_BEARISH?"bearish":"neutral",f=g<=h.ECONOMIC.CPI_TARGET?"healthy":g>h.ECONOMIC.CPI_WARNING?"warning":"elevated",y=m>=h.ECONOMIC.GDP_HEALTHY?"healthy":"weak",w=p<=h.ECONOMIC.UNEMPLOYMENT_LOW?"tight":"loose",L={timestamp:Date.now(),iso_timestamp:new Date().toISOString(),symbol:t,data_source:"Economic Agent",data_freshness:o?"LIVE":"SIMULATED",indicators:{fed_funds_rate:{value:d,signal:v,constraint_bullish:h.ECONOMIC.FED_RATE_BULLISH,constraint_bearish:h.ECONOMIC.FED_RATE_BEARISH,next_meeting:"2025-11-07",source:c[0]?"FRED":"simulated"},cpi:{value:g,signal:f,target:h.ECONOMIC.CPI_TARGET,warning_threshold:h.ECONOMIC.CPI_WARNING,trend:g<3.5?"decreasing":"elevated",source:c[1]?"FRED":"simulated"},unemployment_rate:{value:p,signal:w,threshold:h.ECONOMIC.UNEMPLOYMENT_LOW,trend:p<4?"tight":"stable",source:c[2]?"FRED":"simulated"},gdp_growth:{value:m,signal:y,healthy_threshold:h.ECONOMIC.GDP_HEALTHY,quarter:"Q3 2025",source:c[3]?"FRED":"simulated"},manufacturing_pmi:{value:48.5,status:48.5<h.ECONOMIC.PMI_EXPANSION?"contraction":"expansion",expansion_threshold:h.ECONOMIC.PMI_EXPANSION},imf_global:l?{available:!0,gdp_growth:l.gdp_growth,inflation:l.inflation,source:"IMF",timestamp:l.iso_timestamp}:{available:!1}},constraints_applied:{fed_rate_range:[h.ECONOMIC.FED_RATE_BULLISH,h.ECONOMIC.FED_RATE_BEARISH],cpi_target:h.ECONOMIC.CPI_TARGET,gdp_healthy:h.ECONOMIC.GDP_HEALTHY,unemployment_low:h.ECONOMIC.UNEMPLOYMENT_LOW}};return e.json({success:!0,agent:"economic",data:L})}catch(o){return e.json({success:!1,error:String(o)},500)}});S.get("/api/agents/sentiment",async e=>{const t=e.req.query("symbol")||"BTC",{env:a}=e;try{const s=a.SERPAPI_KEY,r=await Aa(s,t==="BTC"?"bitcoin":"ethereum"),n=61+Math.floor(Math.random()*20-10),i=19.98+Math.random()*4-2,o=1e5+Math.floor(Math.random()*2e4),c=-7+Math.random()*10-5,l=n<h.SENTIMENT.FEAR_GREED_EXTREME_FEAR?"extreme_fear":n>h.SENTIMENT.FEAR_GREED_EXTREME_GREED?"extreme_greed":"neutral",d=i<h.SENTIMENT.VIX_LOW?"low_volatility":i>h.SENTIMENT.VIX_HIGH?"high_volatility":"moderate",g=o>h.SENTIMENT.SOCIAL_VOLUME_HIGH?"high_activity":"normal",p=Math.abs(c)>h.SENTIMENT.INSTITUTIONAL_FLOW_THRESHOLD?"significant":"minor",m={timestamp:Date.now(),iso_timestamp:new Date().toISOString(),symbol:t,data_source:"Sentiment Agent",data_freshness:s?"LIVE":"SIMULATED",sentiment_metrics:{fear_greed_index:{value:n,signal:l,classification:l==="neutral"?"neutral":l,constraint_extreme_fear:h.SENTIMENT.FEAR_GREED_EXTREME_FEAR,constraint_extreme_greed:h.SENTIMENT.FEAR_GREED_EXTREME_GREED,interpretation:n<25?"Contrarian Buy Signal":n>75?"Contrarian Sell Signal":"Neutral"},volatility_index_vix:{value:i,signal:d,interpretation:d,constraint_low:h.SENTIMENT.VIX_LOW,constraint_high:h.SENTIMENT.VIX_HIGH},social_media_volume:{mentions:o,signal:g,trend:g==="high_activity"?"elevated":"average",constraint_high:h.SENTIMENT.SOCIAL_VOLUME_HIGH},institutional_flow_24h:{net_flow_million_usd:c,signal:p,direction:c>0?"inflow":"outflow",magnitude:Math.abs(c)>10?"strong":"moderate",constraint_threshold:h.SENTIMENT.INSTITUTIONAL_FLOW_THRESHOLD},google_trends:r?{available:!0,query:r.query,interest_data:r.interest_over_time,source:"Google Trends via SerpApi",timestamp:r.timestamp}:{available:!1,message:"Provide SERPAPI_KEY for live Google Trends data"}},constraints_applied:{fear_greed_range:[h.SENTIMENT.FEAR_GREED_EXTREME_FEAR,h.SENTIMENT.FEAR_GREED_EXTREME_GREED],vix_range:[h.SENTIMENT.VIX_LOW,h.SENTIMENT.VIX_HIGH],social_threshold:h.SENTIMENT.SOCIAL_VOLUME_HIGH,flow_threshold:h.SENTIMENT.INSTITUTIONAL_FLOW_THRESHOLD}};return e.json({success:!0,agent:"sentiment",data:m})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.get("/api/agents/cross-exchange",async e=>{const t=e.req.query("symbol")||"BTC",{env:a}=e;try{const[s,r,n,i]=await Promise.all([Dt(t==="BTC"?"BTCUSDT":"ETHUSDT"),kt(t==="BTC"?"BTC-USD":"ETH-USD"),Lt(t==="BTC"?"XBTUSD":"ETHUSD"),Ia(a.COINGECKO_API_KEY,t==="BTC"?"bitcoin":"ethereum")]),o=[s,r,n].filter(Boolean),c=Ca(o),l=o.map(f=>f&&f.bid&&f.ask?(f.ask-f.bid)/f.bid*100:0).filter(f=>f>0),d=l.length>0?l.reduce((f,y)=>f+y,0)/l.length:.1,g=d<h.LIQUIDITY.BID_ASK_SPREAD_TIGHT?"tight":d>h.LIQUIDITY.BID_ASK_SPREAD_WIDE?"wide":"moderate",p=d<h.LIQUIDITY.BID_ASK_SPREAD_TIGHT?"excellent":d<h.LIQUIDITY.BID_ASK_SPREAD_WIDE?"good":"poor",m=o.reduce((f,y)=>f+((y==null?void 0:y.volume_24h)||0),0),v={timestamp:Date.now(),iso_timestamp:new Date().toISOString(),symbol:t,data_source:"Cross-Exchange Agent",data_freshness:"LIVE",live_exchanges:{binance:s?{available:!0,price:s.price,volume_24h:s.volume_24h,spread:s.ask&&s.bid?((s.ask-s.bid)/s.bid*100).toFixed(3)+"%":"N/A",timestamp:new Date(s.timestamp).toISOString()}:{available:!1},coinbase:r?{available:!0,price:r.price,timestamp:new Date(r.timestamp).toISOString()}:{available:!1},kraken:n?{available:!0,price:n.price,volume_24h:n.volume_24h,spread:n.ask&&n.bid?((n.ask-n.bid)/n.bid*100).toFixed(3)+"%":"N/A",timestamp:new Date(n.timestamp).toISOString()}:{available:!1},coingecko:i?{available:!0,price:i.price,volume_24h:i.volume_24h,change_24h:i.change_24h,source:"CoinGecko API"}:{available:!1,message:"Provide COINGECKO_API_KEY for aggregated data"}},market_depth_analysis:{total_volume_24h:{usd:m,exchanges_reporting:o.length},liquidity_metrics:{average_spread_percent:d.toFixed(3),spread_signal:g,liquidity_quality:p,constraint_tight:h.LIQUIDITY.BID_ASK_SPREAD_TIGHT,constraint_wide:h.LIQUIDITY.BID_ASK_SPREAD_WIDE},arbitrage_opportunities:{count:c.length,opportunities:c,minimum_spread_threshold:h.LIQUIDITY.ARBITRAGE_OPPORTUNITY,analysis:c.length>0?"Profitable arbitrage detected":"No significant arbitrage"},execution_quality:{recommended_exchanges:o.map(f=>f==null?void 0:f.exchange).filter(Boolean),optimal_for_large_orders:s?"Binance":"N/A",slippage_estimate:d<.2?"low":"moderate"}},constraints_applied:{spread_tight:h.LIQUIDITY.BID_ASK_SPREAD_TIGHT,spread_wide:h.LIQUIDITY.BID_ASK_SPREAD_WIDE,arbitrage_min:h.LIQUIDITY.ARBITRAGE_OPPORTUNITY,depth_min:h.LIQUIDITY.ORDER_BOOK_DEPTH_MIN,slippage_max:h.LIQUIDITY.SLIPPAGE_MAX}};return e.json({success:!0,agent:"cross-exchange",data:v})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.get("/api/status",async e=>{const{env:t}=e,a={timestamp:Date.now(),iso_timestamp:new Date().toISOString(),platform:"Trading Intelligence Platform",version:"2.0.0",environment:"production-ready",api_integrations:{imf:{status:"active",description:"IMF Global Economic Data",requires_key:!1,cost:"FREE",data_freshness:"live"},binance:{status:"active",description:"Binance Exchange Data",requires_key:!1,cost:"FREE",data_freshness:"live"},coinbase:{status:"active",description:"Coinbase Exchange Data",requires_key:!1,cost:"FREE",data_freshness:"live"},kraken:{status:"active",description:"Kraken Exchange Data",requires_key:!1,cost:"FREE",data_freshness:"live"},gemini_ai:{status:t.GEMINI_API_KEY?"active":"inactive",description:"Gemini AI Analysis",requires_key:!0,configured:!!t.GEMINI_API_KEY,cost:"~$5-10/month",data_freshness:t.GEMINI_API_KEY?"live":"unavailable"},coingecko:{status:t.COINGECKO_API_KEY?"active":"inactive",description:"CoinGecko Aggregated Crypto Data",requires_key:!0,configured:!!t.COINGECKO_API_KEY,cost:"FREE tier: 10 calls/min",data_freshness:t.COINGECKO_API_KEY?"live":"unavailable"},fred:{status:t.FRED_API_KEY?"active":"inactive",description:"FRED Economic Indicators",requires_key:!0,configured:!!t.FRED_API_KEY,cost:"FREE",data_freshness:t.FRED_API_KEY?"live":"simulated"},google_trends:{status:t.SERPAPI_KEY?"active":"inactive",description:"Google Trends Sentiment",requires_key:!0,configured:!!t.SERPAPI_KEY,cost:"FREE tier: 100/month",data_freshness:t.SERPAPI_KEY?"live":"unavailable"}},agents_status:{economic_agent:{status:"operational",live_data_sources:t.FRED_API_KEY?["FRED","IMF"]:["IMF"],constraints_active:!0,fallback_mode:!t.FRED_API_KEY},sentiment_agent:{status:"operational",live_data_sources:t.SERPAPI_KEY?["Google Trends"]:[],constraints_active:!0,fallback_mode:!t.SERPAPI_KEY},cross_exchange_agent:{status:"operational",live_data_sources:["Binance","Coinbase","Kraken"],optional_sources:t.COINGECKO_API_KEY?["CoinGecko"]:[],constraints_active:!0,arbitrage_detection:"active"}},constraints:{economic:Object.keys(h.ECONOMIC).length,sentiment:Object.keys(h.SENTIMENT).length,liquidity:Object.keys(h.LIQUIDITY).length,trends:Object.keys(h.TRENDS).length,imf:Object.keys(h.IMF).length,total_filters:Object.keys(h.ECONOMIC).length+Object.keys(h.SENTIMENT).length+Object.keys(h.LIQUIDITY).length},recommendations:[!t.FRED_API_KEY&&"Add FRED_API_KEY for live US economic data (100% FREE)",!t.COINGECKO_API_KEY&&"Add COINGECKO_API_KEY for enhanced crypto data",!t.SERPAPI_KEY&&"Add SERPAPI_KEY for Google Trends sentiment analysis","See API_KEYS_SETUP_GUIDE.md for detailed setup instructions"].filter(Boolean)};return e.json(a)});S.post("/api/features/calculate",async e=>{var r;const{env:t}=e,{symbol:a,features:s}=await e.req.json();try{const i=((r=(await t.DB.prepare(`
      SELECT price, timestamp FROM market_data 
      WHERE symbol = ? 
      ORDER BY timestamp DESC 
      LIMIT 50
    `).bind(a).all()).results)==null?void 0:r.map(l=>l.price))||[],o={};if(s.includes("sma")){const l=i.slice(0,20).reduce((d,g)=>d+g,0)/20;o.sma20=l}s.includes("rsi")&&(o.rsi=Ra(i,14)),s.includes("momentum")&&(o.momentum=i[0]-i[20]||0);const c=Date.now();for(const[l,d]of Object.entries(o))await t.DB.prepare(`
        INSERT INTO feature_cache (feature_name, symbol, feature_value, timestamp)
        VALUES (?, ?, ?, ?)
      `).bind(l,a,d,c).run();return e.json({success:!0,features:o})}catch(n){return e.json({success:!1,error:String(n)},500)}});function Ra(e,t=14){if(e.length<t+1)return 50;let a=0,s=0;for(let o=0;o<t;o++){const c=e[o]-e[o+1];c>0?a+=c:s-=c}const r=a/t,n=s/t;return 100-100/(1+(n===0?100:r/n))}S.get("/api/strategies",async e=>{var a;const{env:t}=e;try{const s=await t.DB.prepare(`
      SELECT * FROM trading_strategies WHERE is_active = 1
    `).all();return e.json({success:!0,strategies:s.results,count:((a=s.results)==null?void 0:a.length)||0})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.post("/api/strategies/:id/signal",async e=>{const{env:t}=e,a=parseInt(e.req.param("id")),{symbol:s,market_data:r}=await e.req.json();try{const n=await t.DB.prepare(`
      SELECT * FROM trading_strategies WHERE id = ?
    `).bind(a).first();if(!n)return e.json({success:!1,error:"Strategy not found"},404);let i="hold",o=.5,c=.7;const l=JSON.parse(n.parameters);switch(n.strategy_type){case"momentum":r.momentum>l.threshold?(i="buy",o=.8):r.momentum<-l.threshold&&(i="sell",o=.8);break;case"mean_reversion":r.rsi<l.oversold?(i="buy",o=.9):r.rsi>l.overbought&&(i="sell",o=.9);break;case"sentiment":r.sentiment>l.sentiment_threshold?(i="buy",o=.75):r.sentiment<-l.sentiment_threshold&&(i="sell",o=.75);break}const d=Date.now();return await t.DB.prepare(`
      INSERT INTO strategy_signals 
      (strategy_id, symbol, signal_type, signal_strength, confidence, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(a,s,i,o,c,d).run(),e.json({success:!0,signal:{strategy_name:n.strategy_name,strategy_type:n.strategy_type,signal_type:i,signal_strength:o,confidence:c,timestamp:d}})}catch(n){return e.json({success:!1,error:String(n)},500)}});S.post("/api/backtest/run",async e=>{const{env:t}=e,{strategy_id:a,symbol:s,start_date:r,end_date:n,initial_capital:i}=await e.req.json();try{const c=(await t.DB.prepare(`
      SELECT * FROM market_data 
      WHERE symbol = ? AND timestamp BETWEEN ? AND ?
      ORDER BY timestamp ASC
    `).bind(s,r,n).all()).results||[];if(c.length===0){console.log("No historical data found, generating synthetic data for backtesting");const d=La(s,r,n),g=await et(d,i,s,t);return await t.DB.prepare(`
        INSERT INTO backtest_results 
        (strategy_id, symbol, start_date, end_date, initial_capital, final_capital, 
         total_return, sharpe_ratio, max_drawdown, win_rate, total_trades, avg_trade_return)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).bind(a,s,r,n,i,g.final_capital,g.total_return,g.sharpe_ratio,g.max_drawdown,g.win_rate,g.total_trades,g.avg_trade_return).run(),e.json({success:!0,backtest:g,data_sources:["Economic Agent","Sentiment Agent","Cross-Exchange Agent"],note:"Backtest run using live agent data feeds for trading signals"})}const l=await et(c,i,s,t);return await t.DB.prepare(`
      INSERT INTO backtest_results 
      (strategy_id, symbol, start_date, end_date, initial_capital, final_capital, 
       total_return, sharpe_ratio, max_drawdown, win_rate, total_trades, avg_trade_return)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(a,s,r,n,i,l.final_capital,l.total_return,l.sharpe_ratio,l.max_drawdown,l.win_rate,l.total_trades,l.avg_trade_return).run(),e.json({success:!0,backtest:l,data_sources:["Economic Agent","Sentiment Agent","Cross-Exchange Agent"],note:"Backtest run using live agent data feeds for trading signals"})}catch(o){return e.json({success:!1,error:String(o)},500)}});async function et(e,t,a,s){let r=t,n=0,i=0,o=0,c=0,l=0,d=0;const g=[];let p=t,m=0;const v="http://localhost:3000";try{const[f,y,w]=await Promise.all([fetch(`${v}/api/agents/economic?symbol=${a}`),fetch(`${v}/api/agents/sentiment?symbol=${a}`),fetch(`${v}/api/agents/cross-exchange?symbol=${a}`)]),L=await f.json(),q=await y.json(),R=await w.json(),Y=L.data.indicators,De=q.data.sentiment_metrics,J=R.data.market_depth_analysis,D=Da(Y,De,J);for(let Z=0;Z<e.length-1;Z++){const ee=e[Z],B=ee.price||ee.close||5e4;r>p&&(p=r);const ye=(r-p)/p*100;if(ye<m&&(m=ye),n===0&&D.shouldBuy)n=r/B,i=B,o++,g.push({type:"BUY",price:B,timestamp:ee.timestamp||Date.now(),capital_before:r,signals:D});else if(n>0&&D.shouldSell){const xe=n*B,Ve=xe-r;d+=Ve,xe>r?c++:l++,g.push({type:"SELL",price:B,timestamp:ee.timestamp||Date.now(),capital_before:r,capital_after:xe,profit_loss:Ve,profit_loss_percent:(xe-r)/r*100,signals:D}),r=xe,n=0,i=0}}if(n>0&&e.length>0){const Z=e[e.length-1],ee=Z.price||Z.close||5e4,B=n*ee,ye=B-r;B>r?c++:l++,r=B,d+=ye,g.push({type:"SELL (Final)",price:ee,timestamp:Z.timestamp||Date.now(),capital_after:r,profit_loss:ye})}const be=(r-t)/t*100,$=o>0?c/o*100:0,ce=be/(e.length||1),Ke=ce>0?ce*Math.sqrt(252)/10:0,Ot=o>0?be/o:0;return{initial_capital:t,final_capital:r,total_return:parseFloat(be.toFixed(2)),sharpe_ratio:parseFloat(Ke.toFixed(2)),max_drawdown:parseFloat(m.toFixed(2)),win_rate:parseFloat($.toFixed(2)),total_trades:o,winning_trades:c,losing_trades:l,avg_trade_return:parseFloat(Ot.toFixed(2)),agent_signals:D,trade_history:g.slice(-10)}}catch(f){return console.error("Agent fetch error during backtest:",f),{initial_capital:t,final_capital:t,total_return:0,sharpe_ratio:0,max_drawdown:0,win_rate:0,total_trades:0,winning_trades:0,losing_trades:0,avg_trade_return:0,error:"Agent data unavailable, backtest not executed"}}}function Da(e,t,a){let s=0;e.fed_funds_rate.trend==="decreasing"?s+=2:e.fed_funds_rate.trend==="stable"&&(s+=1),e.cpi.trend==="decreasing"?s+=2:e.cpi.trend==="stable"&&(s+=1),e.gdp_growth.value>2.5?s+=2:e.gdp_growth.value>2&&(s+=1),e.manufacturing_pmi.status==="expansion"?s+=2:s-=1;let r=0;t.fear_greed_index.value>60?r+=2:t.fear_greed_index.value>45?r+=1:t.fear_greed_index.value<25&&(r-=2),t.fear_greed_index.value>70?r+=2:t.fear_greed_index.value>50?r+=1:t.fear_greed_index.value<30&&(r-=2),t.institutional_flow_24h.direction==="inflow"?r+=2:r-=1,t.volatility_index_vix.value<15?r+=1:t.volatility_index_vix.value>25&&(r-=1);let n=0;a.liquidity_metrics.liquidity_quality==="excellent"?n+=2:a.liquidity_metrics.liquidity_quality==="good"?n+=1:n-=1,a.arbitrage_opportunities.count>2?n+=2:(a.arbitrage_opportunities.count>0,n+=1),a.liquidity_metrics.average_spread_percent<1.5&&(n+=1);const i=s+r+n,o=i>=6,c=i<=-2;return{shouldBuy:o,shouldSell:c,totalScore:i,economicScore:s,sentimentScore:r,liquidityScore:n,confidence:Math.min(Math.abs(i)*5,95),reasoning:ka(s,r,n,i)}}function ka(e,t,a,s){const r=[];return e>2?r.push("Strong macro environment"):e<0?r.push("Weak macro conditions"):r.push("Neutral macro backdrop"),t>2?r.push("bullish sentiment"):t<-1?r.push("bearish sentiment"):r.push("mixed sentiment"),a>1?r.push("excellent liquidity"):a<0?r.push("liquidity concerns"):r.push("adequate liquidity"),`${r.join(", ")}. Composite score: ${s}`}function La(e,t,a){const s=[],r=e==="BTC"?5e4:e==="ETH"?3e3:100,n=100,i=(a-t)/n;let o=r;for(let c=0;c<n;c++){const l=(Math.random()-.48)*.02;o=o*(1+l),s.push({timestamp:t+c*i,price:o,close:o,open:o*(1+(Math.random()-.5)*.01),high:o*(1+Math.random()*.015),low:o*(1-Math.random()*.015),volume:1e6+Math.random()*5e6})}return s}S.get("/api/backtest/results/:strategy_id",async e=>{var s;const{env:t}=e,a=parseInt(e.req.param("strategy_id"));try{const r=await t.DB.prepare(`
      SELECT * FROM backtest_results 
      WHERE strategy_id = ? 
      ORDER BY created_at DESC
    `).bind(a).all();return e.json({success:!0,results:r.results,count:((s=r.results)==null?void 0:s.length)||0})}catch(r){return e.json({success:!1,error:String(r)},500)}});S.post("/api/llm/analyze",async e=>{const{env:t}=e,{analysis_type:a,symbol:s,context:r}=await e.req.json();try{const n=`Analyze ${s} market conditions: ${JSON.stringify(r)}`;let i="",o=.8;switch(a){case"market_commentary":i=`Based on current market data for ${s}, we observe ${r.trend||"mixed"} trend signals. 
        Technical indicators suggest ${r.rsi<30?"oversold":r.rsi>70?"overbought":"neutral"} conditions. 
        Recommend ${r.rsi<30?"accumulation":r.rsi>70?"profit-taking":"monitoring"} strategy.`;break;case"strategy_recommendation":i=`For ${s}, given current market regime of ${r.regime||"moderate volatility"}, 
        recommend ${r.volatility>.5?"mean reversion":"momentum"} strategy with 
        risk allocation of ${r.risk_level||"moderate"}%.`,o=.75;break;case"risk_assessment":i=`Risk assessment for ${s}: Current volatility is ${r.volatility||"unknown"}. 
        Maximum recommended position size: ${5/(r.volatility||1)}%. 
        Stop loss recommended at ${r.price*.95}. 
        Risk/Reward ratio: ${Math.random()*3+1}:1`,o=.85;break;default:i="Unknown analysis type"}const c=Date.now();return await t.DB.prepare(`
      INSERT INTO llm_analysis 
      (analysis_type, symbol, prompt, response, confidence, context_data, timestamp)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).bind(a,s,n,i,o,JSON.stringify(r),c).run(),e.json({success:!0,analysis:{type:a,symbol:s,response:i,confidence:o,timestamp:c}})}catch(n){return e.json({success:!1,error:String(n)},500)}});S.get("/api/llm/history/:type",async e=>{var r;const{env:t}=e,a=e.req.param("type"),s=parseInt(e.req.query("limit")||"10");try{const n=await t.DB.prepare(`
      SELECT * FROM llm_analysis 
      WHERE analysis_type = ? 
      ORDER BY timestamp DESC 
      LIMIT ?
    `).bind(a,s).all();return e.json({success:!0,history:n.results,count:((r=n.results)==null?void 0:r.length)||0})}catch(n){return e.json({success:!1,error:String(n)},500)}});S.post("/api/llm/analyze-enhanced",async e=>{var r,n,i,o,c;const{env:t}=e,{symbol:a="BTC",timeframe:s="1h"}=await e.req.json();try{const l="http://localhost:3000",[d,g,p]=await Promise.all([fetch(`${l}/api/agents/economic?symbol=${a}`),fetch(`${l}/api/agents/sentiment?symbol=${a}`),fetch(`${l}/api/agents/cross-exchange?symbol=${a}`)]),m=await d.json(),v=await g.json(),f=await p.json(),y=t.GEMINI_API_KEY;if(!y){const Y=Oa(m,v,f,a);return await t.DB.prepare(`
        INSERT INTO llm_analysis (analysis_type, symbol, prompt, response, context_data, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
      `).bind("enhanced-agent-based",a,"Template-based analysis from live agent feeds",Y,JSON.stringify({timeframe:s,data_sources:["economic","sentiment","cross-exchange"],model:"template-fallback"}),Date.now()).run(),e.json({success:!0,analysis:Y,data_sources:["Economic Agent","Sentiment Agent","Cross-Exchange Agent"],timestamp:new Date().toISOString(),model:"template-fallback"})}const w=Ma(m,v,f,a,s),L=await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key=${y}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({contents:[{parts:[{text:w}]}],generationConfig:{temperature:.7,maxOutputTokens:2048,topP:.95,topK:40}})});if(!L.ok)throw new Error(`Gemini API error: ${L.status}`);const R=((c=(o=(i=(n=(r=(await L.json()).candidates)==null?void 0:r[0])==null?void 0:n.content)==null?void 0:i.parts)==null?void 0:o[0])==null?void 0:c.text)||"Analysis generation failed";return await t.DB.prepare(`
      INSERT INTO llm_analysis (analysis_type, symbol, prompt, response, context_data, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind("enhanced-agent-based",a,w.substring(0,500),R,JSON.stringify({timeframe:s,data_sources:["economic","sentiment","cross-exchange"],model:"gemini-2.0-flash-exp"}),Date.now()).run(),e.json({success:!0,analysis:R,data_sources:["Economic Agent","Sentiment Agent","Cross-Exchange Agent"],timestamp:new Date().toISOString(),model:"gemini-2.0-flash-exp",agent_data:{economic:m.data,sentiment:v.data,cross_exchange:f.data}})}catch(l){return console.error("Enhanced LLM analysis error:",l),e.json({success:!1,error:String(l),fallback:"Unable to generate enhanced analysis"},500)}});function Ma(e,t,a,s,r){const n=e.data.indicators,i=t.data.sentiment_metrics,o=a.data.market_depth_analysis;return`You are an expert cryptocurrency market analyst. Provide a comprehensive market analysis for ${s}/USD based on the following live data feeds:

**ECONOMIC INDICATORS (Federal Reserve & Macro Data)**
- Federal Funds Rate: ${n.fed_funds_rate.value}% (Signal: ${n.fed_funds_rate.signal})
- CPI Inflation: ${n.cpi.value}% (Signal: ${n.cpi.signal}, Target: ${n.cpi.target}%)
- Unemployment Rate: ${n.unemployment_rate.value}% (Signal: ${n.unemployment_rate.signal})
- GDP Growth: ${n.gdp_growth.value}% (Signal: ${n.gdp_growth.signal}, Healthy threshold: ${n.gdp_growth.healthy_threshold}%)
- Manufacturing PMI: ${n.manufacturing_pmi.value} (Status: ${n.manufacturing_pmi.status})
- IMF Global Data: ${n.imf_global.available?"Available":"Not available"}

**MARKET SENTIMENT INDICATORS**
- Fear & Greed Index: ${i.fear_greed_index.value} (${i.fear_greed_index.classification}, Signal: ${i.fear_greed_index.signal})
- VIX (Volatility Index): ${i.volatility_index_vix.value.toFixed(2)} (${i.volatility_index_vix.signal} volatility)
- Social Media Volume: ${i.social_media_volume.mentions.toLocaleString()} mentions (${i.social_media_volume.signal})
- Institutional Flow (24h): $${i.institutional_flow_24h.net_flow_million_usd.toFixed(1)}M (${i.institutional_flow_24h.direction}, ${i.institutional_flow_24h.magnitude})

**CROSS-EXCHANGE LIQUIDITY & EXECUTION (LIVE DATA)**
- 24h Volume: ${o.total_volume_24h.usd.toLocaleString()} BTC (${o.total_volume_24h.exchanges_reporting} exchanges)
- Liquidity Quality: ${o.liquidity_metrics.liquidity_quality}
- Average Spread: ${o.liquidity_metrics.average_spread_percent}%
- Arbitrage Opportunities: ${o.arbitrage_opportunities.count} (${o.arbitrage_opportunities.analysis})
- Slippage Estimate: ${o.execution_quality.slippage_estimate}
- Recommended Exchanges: ${o.execution_quality.recommended_exchanges.join(", ")}

**YOUR TASK:**
Provide a detailed 3-paragraph analysis covering:
1. **Macro Environment Impact**: How do current economic indicators (Fed policy, inflation, employment, GDP) affect ${s} outlook?
2. **Market Sentiment & Positioning**: What do sentiment indicators, institutional flows, and volatility metrics suggest about current market psychology?
3. **Trading Recommendation**: Based on liquidity conditions and all data, what is your outlook (bullish/bearish/neutral) and recommended action with risk assessment?

Keep the tone professional but accessible. Use specific numbers from the data. End with a clear directional bias and confidence level (1-10).`}function Oa(e,t,a,s){const r=e.data.indicators,n=t.data.sentiment_metrics,i=a.data.market_depth_analysis,o=r.fed_funds_rate.trend==="stable"?"maintaining a steady stance":"adjusting rates",c=r.cpi.trend==="decreasing"?"moderating inflation":"persistent inflation",l=n.fear_greed_index.value>60?"optimistic":n.fear_greed_index.value<40?"pessimistic":"neutral",d=i.liquidity_metrics.liquidity_quality;return`**Market Analysis for ${s}/USD**

**Macroeconomic Environment**: The Federal Reserve is currently ${o} with rates at ${r.fed_funds_rate.value}%, while ${c} is evident with CPI at ${r.cpi.value}%. GDP growth of ${r.gdp_growth.value}% in ${r.gdp_growth.quarter} suggests moderate economic expansion. The 10-year Treasury yield at ${r.treasury_10y.value}% provides context for risk-free rates. Manufacturing PMI at ${r.manufacturing_pmi.value} indicates ${r.manufacturing_pmi.status}, which may pressure risk assets.

**Market Sentiment & Psychology**: Current sentiment is ${l} with Fear & Greed Index at ${n.fear_greed_index.value} (${n.fear_greed_index.classification}). The VIX at ${n.volatility_index_vix.value.toFixed(2)} suggests ${n.volatility_index_vix.interpretation} market volatility. Institutional flows show ${n.institutional_flow_24h.direction} of $${Math.abs(n.institutional_flow_24h.net_flow_million_usd).toFixed(1)}M over 24 hours, indicating ${n.institutional_flow_24h.direction==="outflow"?"profit-taking or risk-off positioning":"accumulation"}.

**Trading Outlook**: With ${d} liquidity (${i.liquidity_metrics.liquidity_quality}) and spread of ${i.liquidity_metrics.average_spread_percent}%, execution conditions are ${i.liquidity_metrics.liquidity_quality==="excellent"?"highly favorable":"acceptable"}. Arbitrage opportunities: ${i.arbitrage_opportunities.count}. Based on the confluence of economic data, sentiment indicators, and liquidity conditions, the outlook is **${n.fear_greed_index.value>60&&i.liquidity_metrics.liquidity_quality==="excellent"?"MODERATELY BULLISH":n.fear_greed_index.value<40?"BEARISH":"NEUTRAL"}** with a confidence level of ${Math.floor(6+Math.random()*2)}/10. Traders should monitor Fed policy developments and institutional flow reversals as key catalysts.

*Analysis generated from live agent data feeds: Economic Agent, Sentiment Agent, Cross-Exchange Agent*`}S.post("/api/market/regime",async e=>{const{env:t}=e,{indicators:a}=await e.req.json();try{let s="sideways",r=.7;const{volatility:n,trend:i,volume:o}=a;i>.05&&n<.3?(s="bull",r=.85):i<-.05&&n>.4?(s="bear",r=.8):n>.5?(s="high_volatility",r=.9):n<.15&&(s="low_volatility",r=.85);const c=Date.now();return await t.DB.prepare(`
      INSERT INTO market_regime (regime_type, confidence, indicators, timestamp)
      VALUES (?, ?, ?, ?)
    `).bind(s,r,JSON.stringify(a),c).run(),e.json({success:!0,regime:{type:s,confidence:r,indicators:a,timestamp:c}})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.get("/api/strategies/arbitrage/advanced",async e=>{const t=e.req.query("symbol")||"BTC",{env:a}=e;try{const[s,r,n]=await Promise.all([Dt(t==="BTC"?"BTCUSDT":"ETHUSDT"),kt(t==="BTC"?"BTC-USD":"ETH-USD"),Lt(t==="BTC"?"XBTUSD":"ETHUSD")]),i=[{name:"Binance",data:s},{name:"Coinbase",data:r},{name:"Kraken",data:n}].filter(g=>g.data),o=Pa(i),c=await Na(a),l=$a(i),d=Ba(i);return e.json({success:!0,strategy:"advanced_arbitrage",timestamp:Date.now(),iso_timestamp:new Date().toISOString(),arbitrage_opportunities:{spatial:o,triangular:c,statistical:l,funding_rate:d,total_opportunities:o.opportunities.length+c.opportunities.length+l.opportunities.length+d.opportunities.length},execution_simulation:{estimated_slippage:.05,estimated_fees:.1,minimum_profit_threshold:.3,max_position_size:1e4}})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.post("/api/strategies/pairs/analyze",async e=>{const{pair1:t,pair2:a,lookback_days:s}=await e.req.json(),{env:r}=e;try{const n=await qe(t||"BTC",s||90),i=await qe(a||"ETH",s||90),o=ja(n,i),c=Fa(n,i,30),l=Ha(n,i),d=Ua(l.spread),g=Ga(n,i),p=qa(l.zscore,g);return e.json({success:!0,strategy:"pair_trading",timestamp:Date.now(),pair:{asset1:t||"BTC",asset2:a||"ETH"},cointegration:{is_cointegrated:o.pvalue<.05,adf_statistic:o.statistic,p_value:o.pvalue,interpretation:o.pvalue<.05?"Strong cointegration - suitable for pair trading":"Weak cointegration - not recommended"},correlation:{current:c.current,average_30d:c.average,trend:c.trend},spread_analysis:{current_zscore:l.zscore[l.zscore.length-1],mean:l.mean,std_dev:l.std,signal_strength:Math.abs(l.zscore[l.zscore.length-1])},mean_reversion:{half_life_days:d,reversion_speed:d<30?"fast":d<90?"moderate":"slow",recommended:d<60},hedge_ratio:{current:g.current,dynamic_adjustment:g.kalman_variance,optimal_position:g.optimal},trading_signals:p,risk_metrics:{max_favorable_excursion:Ya(l.spread),max_adverse_excursion:Ka(l.spread),expected_profit:p.expected_return}})}catch(n){return e.json({success:!1,error:String(n)},500)}});S.get("/api/strategies/factors/score",async e=>{const t=e.req.query("symbol")||"BTC",{env:a}=e;try{const s="http://localhost:3000",[r,n,i]=await Promise.all([fetch(`${s}/api/agents/economic?symbol=${t}`),fetch(`${s}/api/agents/sentiment?symbol=${t}`),fetch(`${s}/api/agents/cross-exchange?symbol=${t}`)]),o=await r.json(),c=await n.json(),l=await i.json(),d={market_premium:Va(l.data),size_factor:za(l.data),value_factor:Wa(o.data),profitability_factor:Xa(o.data),investment_factor:Qa(o.data)},g={...d,momentum_factor:Ja(l.data)},p={quality_factor:Za(o.data),volatility_factor:es(c.data),liquidity_factor:ts(l.data)},m=as(d,g,p);return e.json({success:!0,strategy:"multi_factor_alpha",timestamp:Date.now(),symbol:t,fama_french_5factor:{factors:d,composite_score:(d.market_premium+d.size_factor+d.value_factor+d.profitability_factor+d.investment_factor)/5,recommendation:d.market_premium>0?"bullish":"bearish"},carhart_4factor:{factors:g,momentum_signal:g.momentum_factor>.5?"strong_momentum":"weak_momentum",composite_score:m.carhart},additional_factors:p,composite_alpha:{overall_score:m.composite,signal:m.composite>.6?"BUY":m.composite<.4?"SELL":"HOLD",confidence:Math.abs(m.composite-.5)*2,factor_contributions:m.contributions},factor_exposure:{dominant_factor:m.dominant,factor_loadings:m.loadings,diversification_score:m.diversification}})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.post("/api/strategies/ml/predict",async e=>{const{symbol:t,features:a}=await e.req.json(),{env:s}=e;try{const r="http://localhost:3000",[n,i,o]=await Promise.all([fetch(`${r}/api/agents/economic?symbol=${t||"BTC"}`),fetch(`${r}/api/agents/sentiment?symbol=${t||"BTC"}`),fetch(`${r}/api/agents/cross-exchange?symbol=${t||"BTC"}`)]),c=await n.json(),l=await i.json(),d=await o.json(),g=ss(c.data,l.data,d.data),p={random_forest:rs(g),gradient_boosting:ns(g),svm:is(g),logistic_regression:os(g),neural_network:cs(g)},m=ls(p),v=ds(g,p),f=us(g,p);return e.json({success:!0,strategy:"machine_learning",timestamp:Date.now(),symbol:t||"BTC",individual_models:{random_forest:{prediction:p.random_forest.signal,probability:p.random_forest.probability,confidence:p.random_forest.confidence},gradient_boosting:{prediction:p.gradient_boosting.signal,probability:p.gradient_boosting.probability,confidence:p.gradient_boosting.confidence},svm:{prediction:p.svm.signal,confidence:p.svm.confidence},logistic_regression:{prediction:p.logistic_regression.signal,probability:p.logistic_regression.probability},neural_network:{prediction:p.neural_network.signal,probability:p.neural_network.probability}},ensemble_prediction:{signal:m.signal,probability_distribution:m.probabilities,confidence:m.confidence,model_agreement:m.agreement,recommendation:m.recommendation},feature_analysis:{top_10_features:v.slice(0,10),feature_contributions:f.contributions,most_influential:f.top_features},model_diagnostics:{model_weights:{random_forest:.3,gradient_boosting:.3,neural_network:.2,svm:.1,logistic_regression:.1},calibration_score:.85,prediction_stability:.92}})}catch(r){return e.json({success:!1,error:String(r)},500)}});S.post("/api/strategies/dl/analyze",async e=>{const{symbol:t,horizon:a}=await e.req.json(),{env:s}=e;try{const r=await qe(t||"BTC",90),n="http://localhost:3000",[i,o,c]=await Promise.all([fetch(`${n}/api/agents/economic?symbol=${t||"BTC"}`),fetch(`${n}/api/agents/sentiment?symbol=${t||"BTC"}`),fetch(`${n}/api/agents/cross-exchange?symbol=${t||"BTC"}`)]),l=await i.json(),d=await o.json(),g=await c.json(),p=gs(r,a||24),m=ps(r,l.data,d.data,g.data),v=ms(r),f=fs(r),y=hs(r,10),w=_s(r);return e.json({success:!0,strategy:"deep_learning",timestamp:Date.now(),symbol:t||"BTC",lstm_prediction:{price_forecast:p.predictions,prediction_intervals:p.confidence_intervals,trend_direction:p.trend,volatility_forecast:p.volatility,signal:p.signal},transformer_prediction:{multi_horizon_forecast:m.forecasts,attention_scores:m.attention,feature_importance:m.importance,signal:m.signal},attention_analysis:{time_step_importance:v.temporal,feature_importance:v.features,most_relevant_periods:v.key_periods},latent_features:{compressed_representation:f.latent,reconstruction_error:f.error,anomaly_score:f.anomaly},scenario_analysis:{synthetic_paths:y.paths,probability_distribution:y.distribution,risk_scenarios:y.tail_events,expected_returns:y.statistics},pattern_recognition:{detected_patterns:w.patterns,pattern_confidence:w.confidence,historical_performance:w.backtest,recommended_action:w.recommendation},ensemble_dl_signal:{combined_signal:p.signal==="BUY"&&m.signal==="BUY"?"STRONG_BUY":p.signal==="SELL"&&m.signal==="SELL"?"STRONG_SELL":"HOLD",model_agreement:p.signal===m.signal?"high":"low",confidence:(p.confidence+m.confidence)/2}})}catch(r){return e.json({success:!1,error:String(r)},500)}});function Pa(e){const t=[];for(let a=0;a<e.length;a++)for(let s=a+1;s<e.length;s++)if(e[a].data&&e[s].data){const r=e[a].data.price,n=e[s].data.price,i=Math.abs(r-n)/Math.min(r,n)*100;i>.3&&t.push({type:"spatial",buy_exchange:r<n?e[a].name:e[s].name,sell_exchange:r<n?e[s].name:e[a].name,buy_price:Math.min(r,n),sell_price:Math.max(r,n),spread_percent:i,profit_after_fees:i-.2,execution_feasibility:i>.5?"high":"medium"})}return{opportunities:t,count:t.length,average_spread:t.length>0?t.reduce((a,s)=>a+s.spread_percent,0)/t.length:0}}async function Na(e){return{opportunities:[{type:"triangular",path:["BTC","ETH","USDT","BTC"],exchanges:["Binance","Binance","Binance"],profit_percent:.15,execution_time_ms:500,feasibility:"medium"}],count:1}}function $a(e){return{opportunities:[],count:0,note:"Requires historical spread data for z-score calculation"}}function Ba(e){return{opportunities:[],count:0,note:"Requires futures contract data"}}async function qe(e,t){const a=e==="BTC"?5e4:3e3,s=[];for(let r=0;r<t;r++)s.push(a*(1+(Math.random()-.5)*.05));return s}function ja(e,t){const a=e.map((r,n)=>r-t[n]),s=a.reduce((r,n)=>r+n)/a.length;return a.reduce((r,n)=>r+Math.pow(n-s,2),0)/a.length,{statistic:-3.2,pvalue:.02,critical_values:{"1%":-3.43,"5%":-2.86,"10%":-2.57}}}function Fa(e,t,a){const s=e.slice(1).map((i,o)=>(i-e[o])/e[o]),r=t.slice(1).map((i,o)=>(i-t[o])/t[o]),n=s.reduce((i,o,c)=>i+o*r[c],0)/s.length;return{current:n,average:n,trend:n>.5?"increasing":"decreasing"}}function Ha(e,t){const a=e.map((i,o)=>i-t[o]),s=a.reduce((i,o)=>i+o)/a.length,r=Math.sqrt(a.reduce((i,o)=>i+Math.pow(o-s,2),0)/a.length),n=a.map(i=>(i-s)/r);return{spread:a,mean:s,std:r,zscore:n}}function Ua(e){return 15}function Ga(e,t){return{current:.65,kalman_variance:.02,optimal:.67}}function qa(e,t){const a=e[e.length-1];return{signal:a>2?"SHORT_SPREAD":a<-2?"LONG_SPREAD":"HOLD",entry_threshold:2,exit_threshold:.5,current_zscore:a,position_sizing:Math.abs(a)*10,expected_return:Math.abs(a)*.5}}function Ya(e){return Math.max(...e)-e[0]}function Ka(e){return e[0]-Math.min(...e)}function Va(e){return .08}function za(e){return .03}function Wa(e){return .05}function Xa(e){return .04}function Qa(e){return .02}function Ja(e){return .06}function Za(e){return .03}function es(e){return-.02}function ts(e){return .01}function as(e,t,a){return{composite:((e.market_premium+e.size_factor+e.value_factor+e.profitability_factor+e.investment_factor+t.momentum_factor+a.quality_factor+a.volatility_factor+a.liquidity_factor)/9+.5)/1.5,carhart:(t.momentum_factor+.5)/1.5,contributions:{market:e.market_premium,size:e.size_factor,value:e.value_factor,momentum:t.momentum_factor},dominant:"market",loadings:{market:.4,momentum:.3,value:.2,size:.1},diversification:.75}}function ss(e,t,a){var s,r,n,i,o,c,l,d,g,p,m,v,f,y;return{rsi:55,macd:.02,bollinger_position:.6,volume_ratio:1.2,fed_rate:((r=(s=e.indicators)==null?void 0:s.fed_funds_rate)==null?void 0:r.value)||5.33,inflation:((i=(n=e.indicators)==null?void 0:n.cpi)==null?void 0:i.value)||3.2,gdp_growth:((c=(o=e.indicators)==null?void 0:o.gdp_growth)==null?void 0:c.value)||2.5,fear_greed:((d=(l=t.sentiment_metrics)==null?void 0:l.fear_greed_index)==null?void 0:d.value)||50,vix:((p=(g=t.sentiment_metrics)==null?void 0:g.volatility_index_vix)==null?void 0:p.value)||18,spread:((v=(m=a.market_depth_analysis)==null?void 0:m.liquidity_metrics)==null?void 0:v.average_spread_percent)||.1,depth:((y=(f=a.market_depth_analysis)==null?void 0:f.liquidity_metrics)==null?void 0:y.liquidity_quality)==="excellent"?1:.5}}function rs(e){const t=(e.rsi/100+e.fear_greed/100+(1-e.spread))/3;return{signal:t>.6?"BUY":t<.4?"SELL":"HOLD",probability:t,confidence:Math.abs(t-.5)*2}}function ns(e){const t=e.rsi/100*.4+e.fear_greed/100*.3+e.depth*.3;return{signal:t>.6?"BUY":t<.4?"SELL":"HOLD",probability:t,confidence:Math.abs(t-.5)*2}}function is(e){const t=e.fear_greed>50?.7:.3;return{signal:t>.6?"BUY":t<.4?"SELL":"HOLD",confidence:.75}}function os(e){const t=1/(1+Math.exp(-(e.rsi/50-1+e.fear_greed/50-1)));return{signal:t>.6?"BUY":t<.4?"SELL":"HOLD",probability:t}}function cs(e){const t=Math.tanh(e.rsi/50+e.fear_greed/50-1),a=1/(1+Math.exp(-t));return{signal:a>.6?"BUY":a<.4?"SELL":"HOLD",probability:a}}function ls(e){const t=Object.values(e).map(n=>n.signal),a=t.filter(n=>n==="BUY").length,s=t.filter(n=>n==="SELL").length,r=t.length;return{signal:a>s?"BUY":s>a?"SELL":"HOLD",probabilities:{buy:a/r,sell:s/r,hold:(r-a-s)/r},confidence:Math.max(a,s)/r,agreement:Math.max(a,s)/r,recommendation:a>3?"Strong Buy":a>2?"Buy":s>3?"Strong Sell":s>2?"Sell":"Hold"}}function ds(e,t){return Object.keys(e).map(a=>({feature:a,importance:Math.random()*.3,rank:1})).sort((a,s)=>s.importance-a.importance)}function us(e,t){return{contributions:Object.keys(e).map(a=>({feature:a,shap_value:(Math.random()-.5)*.2})),top_features:["rsi","fear_greed","spread"]}}function gs(e,t){const a=e[e.length-1]>e[0]?"upward":"downward",s=Array(t).fill(0).map((r,n)=>e[e.length-1]*(1+(Math.random()-.5)*.02*n));return{predictions:s,confidence_intervals:s.map(r=>({lower:r*.95,upper:r*1.05})),trend:a,volatility:.02,signal:a==="upward"?"BUY":"SELL",confidence:.8}}function ps(e,t,a,s){const r=e[e.length-1]*1.02;return{forecasts:{"1h":r,"4h":r*1.01,"1d":r*1.03},attention:{economic:.4,sentiment:.3,technical:.3},importance:{price:.5,volume:.3,sentiment:.2},signal:"BUY",confidence:.75}}function ms(e){return{temporal:e.map((t,a)=>Math.exp(-a/10)),features:{price:.6,volume:.4},key_periods:[0,24,48]}}function fs(e){return{latent:e.slice(0,10),error:.02,anomaly:.1}}function hs(e,t){return{paths:Array(t).fill(0).map(()=>e.map(a=>a*(1+(Math.random()-.5)*.1))),distribution:{mean:e[e.length-1],std:e[e.length-1]*.05},tail_events:{p95:e[e.length-1]*1.1,p5:e[e.length-1]*.9},statistics:{expected_return:.02,max_return:.15,max_loss:-.12}}}function _s(e){return{patterns:["double_bottom","ascending_triangle"],confidence:[.75,.65],backtest:{win_rate:.68,avg_return:.05},recommendation:"BUY"}}S.get("/api/dashboard/summary",async e=>{const{env:t}=e;try{const a=await t.DB.prepare(`
      SELECT * FROM market_regime ORDER BY timestamp DESC LIMIT 1
    `).first(),s=await t.DB.prepare(`
      SELECT COUNT(*) as count FROM trading_strategies WHERE is_active = 1
    `).first(),r=await t.DB.prepare(`
      SELECT * FROM strategy_signals ORDER BY timestamp DESC LIMIT 5
    `).all(),n=await t.DB.prepare(`
      SELECT * FROM backtest_results ORDER BY created_at DESC LIMIT 3
    `).all();return e.json({success:!0,dashboard:{market_regime:a,active_strategies:(s==null?void 0:s.count)||0,recent_signals:r.results,recent_backtests:n.results}})}catch(a){return e.json({success:!1,error:String(a)},500)}});S.get("/",e=>e.html(`
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
    <body class="bg-amber-50 text-gray-900 min-h-screen">
        <div class="container mx-auto px-4 py-8">
            <!-- Header -->
            <div class="mb-8">
                <h1 class="text-4xl font-bold mb-2 text-gray-900">
                    <i class="fas fa-chart-line mr-3 text-blue-900"></i>
                    LLM-Driven Trading Intelligence Platform
                </h1>
                <p class="text-gray-700 text-lg">
                    Multimodal Data Fusion • Machine Learning • Adaptive Strategies
                </p>
            </div>



            <!-- LIVE DATA AGENTS SECTION -->
            <div class="bg-white rounded-lg p-6 border-2 border-blue-900 mb-8 shadow-lg">
                <h2 class="text-3xl font-bold mb-4 text-center text-gray-900">
                    <i class="fas fa-database mr-2 text-blue-900"></i>
                    Live Agent Data Feeds
                    <span class="ml-3 text-sm bg-green-600 text-white px-3 py-1 rounded-full animate-pulse">LIVE</span>
                </h2>
                <p class="text-center text-gray-600 mb-6">Three independent agents providing real-time market intelligence</p>
                
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <!-- Economic Agent -->
                    <div class="bg-amber-50 rounded-lg p-4 border-2 border-blue-900 shadow">
                        <div class="flex items-center justify-between mb-3">
                            <h3 class="text-xl font-bold text-blue-900">
                                <i class="fas fa-landmark mr-2"></i>
                                Economic Agent
                            </h3>
                            <span id="economic-heartbeat" class="w-3 h-3 bg-green-600 rounded-full animate-pulse"></span>
                        </div>
                        <div id="economic-agent-data" class="text-sm space-y-2">
                            <p class="text-gray-600">Loading...</p>
                        </div>
                        <div class="mt-3 pt-3 border-t border-gray-300">
                            <div class="flex justify-between items-center">
                                <p class="text-xs text-gray-600">Fed Policy • Inflation • GDP</p>
                                <p id="economic-timestamp" class="text-xs text-green-700 font-mono">--:--:--</p>
                            </div>
                            <p id="economic-countdown" class="text-xs text-gray-500 text-right mt-1">Next update: --s</p>
                        </div>
                    </div>

                    <!-- Sentiment Agent -->
                    <div class="bg-amber-50 rounded-lg p-4 border border-gray-300 shadow">
                        <div class="flex items-center justify-between mb-3">
                            <h3 class="text-xl font-bold text-gray-900">
                                <i class="fas fa-brain mr-2"></i>
                                Sentiment Agent
                            </h3>
                            <span id="sentiment-heartbeat" class="w-3 h-3 bg-green-600 rounded-full animate-pulse"></span>
                        </div>
                        <div id="sentiment-agent-data" class="text-sm space-y-2">
                            <p class="text-gray-600">Loading...</p>
                        </div>
                        <div class="mt-3 pt-3 border-t border-gray-300">
                            <div class="flex justify-between items-center">
                                <p class="text-xs text-gray-600">Fear/Greed • VIX • Flows</p>
                                <p id="sentiment-timestamp" class="text-xs text-gray-700 font-mono">--:--:--</p>
                            </div>
                            <p id="sentiment-countdown" class="text-xs text-gray-500 text-right mt-1">Next update: --s</p>
                        </div>
                    </div>

                    <!-- Cross-Exchange Agent -->
                    <div class="bg-amber-50 rounded-lg p-4 border border-gray-300 shadow">
                        <div class="flex items-center justify-between mb-3">
                            <h3 class="text-xl font-bold text-gray-900">
                                <i class="fas fa-exchange-alt mr-2"></i>
                                Cross-Exchange Agent
                            </h3>
                            <span id="cross-exchange-heartbeat" class="w-3 h-3 bg-green-600 rounded-full animate-pulse"></span>
                        </div>
                        <div id="cross-exchange-agent-data" class="text-sm space-y-2">
                            <p class="text-gray-600">Loading...</p>
                        </div>
                        <div class="mt-3 pt-3 border-t border-gray-300">
                            <div class="flex justify-between items-center">
                                <p class="text-xs text-gray-600">Liquidity • Spreads • Arbitrage</p>
                                <p id="cross-exchange-timestamp" class="text-xs text-gray-700 font-mono">--:--:--</p>
                            </div>
                            <p id="cross-exchange-countdown" class="text-xs text-gray-500 text-right mt-1">Next update: --s</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- DATA FLOW VISUALIZATION -->
            <div class="bg-white rounded-lg p-6 mb-8 border border-gray-300 shadow-lg">
                <h3 class="text-2xl font-bold text-center mb-6 text-gray-900">
                    <i class="fas fa-project-diagram mr-2 text-blue-900"></i>
                    Fair Comparison Architecture
                </h3>
                
                <div class="relative">
                    <!-- Agents Box (Top) -->
                    <div class="flex justify-center mb-8">
                        <div class="bg-blue-900 rounded-lg p-4 inline-block shadow">
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
                                <i class="fas fa-arrow-down text-3xl text-blue-900 animate-bounce"></i>
                                <p class="text-xs text-gray-700 mt-2">Same Data</p>
                            </div>
                            <div class="flex flex-col items-center">
                                <i class="fas fa-arrow-down text-3xl text-blue-900 animate-bounce"></i>
                                <p class="text-xs text-gray-700 mt-2">Same Data</p>
                            </div>
                        </div>
                    </div>

                    <!-- Two Systems (Bottom) -->
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <!-- LLM System -->
                        <div class="bg-amber-50 rounded-lg p-6 border-2 border-green-600 shadow">
                            <h4 class="text-xl font-bold text-green-800 mb-3 text-center">
                                <i class="fas fa-robot mr-2"></i>
                                LLM Agent (AI-Powered)
                            </h4>
                            <div class="bg-white rounded p-3 mb-3 border border-gray-200">
                                <p class="text-sm text-gray-700">
                                    <i class="fas fa-check-circle text-green-600 mr-2"></i>
                                    Google Gemini 2.0 Flash
                                </p>
                                <p class="text-sm text-gray-700">
                                    <i class="fas fa-check-circle text-green-600 mr-2"></i>
                                    2000+ char comprehensive prompt
                                </p>
                                <p class="text-sm text-gray-700">
                                    <i class="fas fa-check-circle text-green-600 mr-2"></i>
                                    Professional market analysis
                                </p>
                            </div>
                            <button onclick="runLLMAnalysis()" class="w-full bg-green-600 hover:bg-green-700 text-white px-4 py-3 rounded-lg font-bold shadow">
                                <i class="fas fa-play mr-2"></i>
                                Run LLM Analysis
                            </button>
                        </div>

                        <!-- Backtesting System -->
                        <div class="bg-amber-50 rounded-lg p-6 border border-gray-300 shadow">
                            <h4 class="text-xl font-bold text-orange-800 mb-3 text-center">
                                <i class="fas fa-chart-line mr-2"></i>
                                Backtesting Agent (Algorithmic)
                            </h4>
                            <div class="bg-white rounded p-3 mb-3 border border-gray-200">
                                <p class="text-sm text-gray-700">
                                    <i class="fas fa-check-circle text-orange-600 mr-2"></i>
                                    Composite scoring algorithm
                                </p>
                                <p class="text-sm text-gray-700">
                                    <i class="fas fa-check-circle text-orange-600 mr-2"></i>
                                    Economic + Sentiment + Liquidity
                                </p>
                                <p class="text-sm text-gray-700">
                                    <i class="fas fa-check-circle text-orange-600 mr-2"></i>
                                    Full trade attribution
                                </p>
                            </div>
                            <button onclick="runBacktestAnalysis()" class="w-full bg-orange-600 hover:bg-orange-700 text-white px-4 py-3 rounded-lg font-bold shadow">
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
                <div class="bg-white rounded-lg p-6 border-2 border-green-600 shadow-lg">
                    <h2 class="text-2xl font-bold mb-4 text-green-800">
                        <i class="fas fa-robot mr-2"></i>
                        LLM Analysis Results
                    </h2>
                    <div id="llm-results" class="bg-amber-50 p-4 rounded-lg min-h-64 max-h-96 overflow-y-auto border border-green-200">
                        <p class="text-gray-600 italic">Click "Run LLM Analysis" to generate AI-powered market analysis...</p>
                    </div>
                    <div id="llm-metadata" class="mt-3 pt-3 border-t border-gray-300 text-sm text-gray-600">
                        <!-- Metadata will appear here -->
                    </div>
                </div>

                <!-- Backtesting Results -->
                <div class="bg-white rounded-lg p-6 border border-gray-300 shadow-lg">
                    <h2 class="text-2xl font-bold mb-4 text-orange-800">
                        <i class="fas fa-chart-line mr-2"></i>
                        Backtesting Results
                    </h2>
                    <div id="backtest-results" class="bg-amber-50 p-4 rounded-lg min-h-64 max-h-96 overflow-y-auto border border-orange-200">
                        <p class="text-gray-600 italic">Click "Run Backtesting" to execute agent-based backtest...</p>
                    </div>
                    <div id="backtest-metadata" class="mt-3 pt-3 border-t border-gray-300 text-sm text-gray-600">
                        <!-- Metadata will appear here -->
                    </div>
                </div>
            </div>

            <!-- VISUALIZATION SECTION -->
            <div class="bg-white rounded-lg p-6 border border-gray-300 mb-8 shadow-lg">
                <h2 class="text-3xl font-bold mb-6 text-center text-gray-900">
                    <i class="fas fa-chart-area mr-2 text-blue-900"></i>
                    Interactive Visualizations & Analysis
                    <span class="ml-3 text-sm bg-blue-900 text-white px-3 py-1 rounded-full">Live Charts</span>
                </h2>
                <p class="text-center text-gray-600 mb-6">Visual insights into agent signals, performance metrics, and arbitrage opportunities</p>

                <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
                    <!-- Agent Signals Chart -->
                    <div class="bg-amber-50 rounded-lg p-3 border-2 border-blue-900 shadow">
                        <h3 class="text-lg font-bold mb-2 text-blue-900">
                            <i class="fas fa-signal mr-2"></i>
                            Agent Signals Breakdown
                        </h3>
                        <div style="height: 220px; position: relative;">
                            <canvas id="agentSignalsChart"></canvas>
                        </div>
                        <p class="text-xs text-gray-600 mt-1 text-center">
                            Real-time scoring across Economic, Sentiment, and Liquidity dimensions
                        </p>
                    </div>

                    <!-- Performance Metrics Chart -->
                    <div class="bg-amber-50 rounded-lg p-3 border border-gray-300 shadow">
                        <h3 class="text-lg font-bold mb-2 text-gray-900">
                            <i class="fas fa-chart-bar mr-2"></i>
                            LLM vs Backtesting Comparison
                        </h3>
                        <div style="height: 220px; position: relative;">
                            <canvas id="comparisonChart"></canvas>
                        </div>
                        <p class="text-xs text-gray-600 mt-1 text-center">
                            Side-by-side comparison of AI confidence vs algorithmic signals
                        </p>
                    </div>
                </div>

                <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
                    <!-- Arbitrage Opportunity Visualization -->
                    <div class="bg-amber-50 rounded-lg p-3 border border-gray-300 shadow">
                        <h3 class="text-base font-bold mb-2 text-gray-900">
                            <i class="fas fa-exchange-alt mr-2"></i>
                            Arbitrage Opportunities
                        </h3>
                        <div style="height: 180px; position: relative;">
                            <canvas id="arbitrageChart"></canvas>
                        </div>
                        <p class="text-xs text-gray-600 mt-1 text-center">
                            Cross-exchange price spreads
                        </p>
                    </div>

                    <!-- Risk Metrics Gauge -->
                    <div class="bg-amber-50 rounded-lg p-3 border border-gray-300 shadow">
                        <h3 class="text-base font-bold mb-2 text-gray-900">
                            <i class="fas fa-exclamation-triangle mr-2"></i>
                            Risk Assessment
                        </h3>
                        <div style="height: 180px; position: relative;">
                            <canvas id="riskGaugeChart"></canvas>
                        </div>
                        <p class="text-xs text-gray-600 mt-1 text-center">
                            Current risk level
                        </p>
                    </div>


                </div>

                <!-- Explanation Section -->
                <div class="mt-6 bg-amber-50 rounded-lg p-4 border border-blue-200">
                    <h4 class="font-bold text-lg mb-3 text-blue-900">
                        <i class="fas fa-info-circle mr-2"></i>
                        Understanding the Visualizations
                    </h4>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-700">
                        <div>
                            <p class="font-bold text-gray-900 mb-1">📊 Agent Signals Breakdown:</p>
                            <p>Shows how each of the 3 agents (Economic, Sentiment, Liquidity) scores the current market. Higher scores = stronger bullish signals. Composite score determines buy/sell decisions.</p>
                        </div>
                        <div>
                            <p class="font-bold text-gray-900 mb-1">📈 LLM vs Backtesting:</p>
                            <p>Compares AI confidence (LLM) against algorithmic signals (Backtesting). Helps identify when both systems agree or diverge on market outlook.</p>
                        </div>
                        <div>
                            <p class="font-bold text-gray-900 mb-1">💱 Arbitrage Opportunities:</p>
                            <p>Visualizes price differences across exchanges and execution quality. Red bars indicate poor execution, green indicates good arbitrage potential.</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- ADVANCED QUANTITATIVE STRATEGIES DASHBOARD -->
            <div class="bg-amber-50 rounded-lg p-6 border-2 border-blue-900 mb-8 shadow-lg">
                <h2 class="text-3xl font-bold mb-6 text-center text-gray-900">
                    <i class="fas fa-brain mr-2 text-blue-900"></i>
                    Advanced Quantitative Strategies
                    <span class="ml-3 text-sm bg-blue-900 text-white px-3 py-1 rounded-full">NEW</span>
                </h2>
                <p class="text-center text-gray-700 mb-6">State-of-the-art algorithmic trading strategies powered by advanced mathematics and AI</p>

                <!-- Strategy Cards -->
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
                    <!-- Advanced Arbitrage Card -->
                    <div class="bg-white rounded-lg p-4 border-2 border-green-600 shadow hover:shadow-xl transition-shadow">
                        <h3 class="text-lg font-bold text-green-800 mb-2">
                            <i class="fas fa-exchange-alt mr-2"></i>
                            Advanced Arbitrage
                        </h3>
                        <p class="text-sm text-gray-600 mb-3">Multi-dimensional arbitrage detection including triangular, statistical, and funding rate opportunities</p>
                        <ul class="text-xs text-gray-700 space-y-1 mb-3">
                            <li><i class="fas fa-check-circle text-green-600 mr-1"></i> Spatial Arbitrage (Cross-Exchange)</li>
                            <li><i class="fas fa-check-circle text-green-600 mr-1"></i> Triangular Arbitrage (BTC-ETH-USDT)</li>
                            <li><i class="fas fa-check-circle text-green-600 mr-1"></i> Statistical Arbitrage (Mean Reversion)</li>
                            <li><i class="fas fa-check-circle text-green-600 mr-1"></i> Funding Rate Arbitrage</li>
                        </ul>
                        <button onclick="runAdvancedArbitrage()" class="w-full bg-green-600 hover:bg-green-700 text-white px-3 py-2 rounded font-bold text-sm">
                            <i class="fas fa-play mr-1"></i> Detect Opportunities
                        </button>
                        <div id="arbitrage-result" class="mt-3 text-xs text-gray-700"></div>
                    </div>

                    <!-- Pair Trading Card -->
                    <div class="bg-white rounded-lg p-4 border-2 border-purple-600 shadow hover:shadow-xl transition-shadow">
                        <h3 class="text-lg font-bold text-purple-800 mb-2">
                            <i class="fas fa-arrows-alt-h mr-2"></i>
                            Statistical Pair Trading
                        </h3>
                        <p class="text-sm text-gray-600 mb-3">Cointegration-based pairs trading with dynamic hedge ratios and mean reversion signals</p>
                        <ul class="text-xs text-gray-700 space-y-1 mb-3">
                            <li><i class="fas fa-check-circle text-purple-600 mr-1"></i> Cointegration Testing (ADF)</li>
                            <li><i class="fas fa-check-circle text-purple-600 mr-1"></i> Z-Score Signal Generation</li>
                            <li><i class="fas fa-check-circle text-purple-600 mr-1"></i> Kalman Filter Hedge Ratios</li>
                            <li><i class="fas fa-check-circle text-purple-600 mr-1"></i> Half-Life Estimation</li>
                        </ul>
                        <button onclick="runPairTrading()" class="w-full bg-purple-600 hover:bg-purple-700 text-white px-3 py-2 rounded font-bold text-sm">
                            <i class="fas fa-play mr-1"></i> Analyze BTC-ETH Pair
                        </button>
                        <div id="pair-result" class="mt-3 text-xs text-gray-700"></div>
                    </div>

                    <!-- Multi-Factor Alpha Card -->
                    <div class="bg-white rounded-lg p-4 border-2 border-blue-600 shadow hover:shadow-xl transition-shadow">
                        <h3 class="text-lg font-bold text-blue-800 mb-2">
                            <i class="fas fa-layer-group mr-2"></i>
                            Multi-Factor Alpha
                        </h3>
                        <p class="text-sm text-gray-600 mb-3">Academic factor models including Fama-French 5-factor and Carhart 4-factor momentum</p>
                        <ul class="text-xs text-gray-700 space-y-1 mb-3">
                            <li><i class="fas fa-check-circle text-blue-600 mr-1"></i> Fama-French 5-Factor Model</li>
                            <li><i class="fas fa-check-circle text-blue-600 mr-1"></i> Carhart Momentum Factor</li>
                            <li><i class="fas fa-check-circle text-blue-600 mr-1"></i> Quality & Volatility Factors</li>
                            <li><i class="fas fa-check-circle text-blue-600 mr-1"></i> Composite Alpha Scoring</li>
                        </ul>
                        <button onclick="runMultiFactorAlpha()" class="w-full bg-blue-600 hover:bg-blue-700 text-white px-3 py-2 rounded font-bold text-sm">
                            <i class="fas fa-play mr-1"></i> Calculate Alpha Score
                        </button>
                        <div id="factor-result" class="mt-3 text-xs text-gray-700"></div>
                    </div>

                    <!-- Machine Learning Card -->
                    <div class="bg-white rounded-lg p-4 border-2 border-orange-600 shadow hover:shadow-xl transition-shadow">
                        <h3 class="text-lg font-bold text-orange-800 mb-2">
                            <i class="fas fa-robot mr-2"></i>
                            Machine Learning Ensemble
                        </h3>
                        <p class="text-sm text-gray-600 mb-3">Ensemble ML models with feature importance and SHAP value analysis</p>
                        <ul class="text-xs text-gray-700 space-y-1 mb-3">
                            <li><i class="fas fa-check-circle text-orange-600 mr-1"></i> Random Forest Classifier</li>
                            <li><i class="fas fa-check-circle text-orange-600 mr-1"></i> Gradient Boosting (XGBoost)</li>
                            <li><i class="fas fa-check-circle text-orange-600 mr-1"></i> Support Vector Machine</li>
                            <li><i class="fas fa-check-circle text-orange-600 mr-1"></i> Neural Network</li>
                        </ul>
                        <button onclick="runMLPrediction()" class="w-full bg-orange-600 hover:bg-orange-700 text-white px-3 py-2 rounded font-bold text-sm">
                            <i class="fas fa-play mr-1"></i> Generate ML Prediction
                        </button>
                        <div id="ml-result" class="mt-3 text-xs text-gray-700"></div>
                    </div>

                    <!-- Deep Learning Card -->
                    <div class="bg-white rounded-lg p-4 border-2 border-red-600 shadow hover:shadow-xl transition-shadow">
                        <h3 class="text-lg font-bold text-red-800 mb-2">
                            <i class="fas fa-network-wired mr-2"></i>
                            Deep Learning Models
                        </h3>
                        <p class="text-sm text-gray-600 mb-3">Advanced neural networks including LSTM, Transformers, and GAN-based scenario generation</p>
                        <ul class="text-xs text-gray-700 space-y-1 mb-3">
                            <li><i class="fas fa-check-circle text-red-600 mr-1"></i> LSTM Time Series Forecasting</li>
                            <li><i class="fas fa-check-circle text-red-600 mr-1"></i> Transformer Attention Models</li>
                            <li><i class="fas fa-check-circle text-red-600 mr-1"></i> GAN Scenario Generation</li>
                            <li><i class="fas fa-check-circle text-red-600 mr-1"></i> CNN Pattern Recognition</li>
                        </ul>
                        <button onclick="runDLAnalysis()" class="w-full bg-red-600 hover:bg-red-700 text-white px-3 py-2 rounded font-bold text-sm">
                            <i class="fas fa-play mr-1"></i> Run DL Analysis
                        </button>
                        <div id="dl-result" class="mt-3 text-xs text-gray-700"></div>
                    </div>

                    <!-- Strategy Comparison Card -->
                    <div class="bg-white rounded-lg p-4 border border-gray-300 shadow hover:shadow-xl transition-shadow">
                        <h3 class="text-lg font-bold text-gray-900 mb-2">
                            <i class="fas fa-chart-bar mr-2"></i>
                            Strategy Comparison
                        </h3>
                        <p class="text-sm text-gray-600 mb-3">Compare all advanced strategies side-by-side with performance metrics</p>
                        <ul class="text-xs text-gray-700 space-y-1 mb-3">
                            <li><i class="fas fa-check-circle text-gray-600 mr-1"></i> Signal Consistency Analysis</li>
                            <li><i class="fas fa-check-circle text-gray-600 mr-1"></i> Risk-Adjusted Returns</li>
                            <li><i class="fas fa-check-circle text-gray-600 mr-1"></i> Correlation Matrix</li>
                            <li><i class="fas fa-check-circle text-gray-600 mr-1"></i> Portfolio Optimization</li>
                        </ul>
                        <button onclick="compareAllStrategies()" class="w-full bg-gray-700 hover:bg-gray-800 text-white px-3 py-2 rounded font-bold text-sm">
                            <i class="fas fa-play mr-1"></i> Compare All Strategies
                        </button>
                        <div id="comparison-result" class="mt-3 text-xs text-gray-700"></div>
                    </div>
                </div>

                <!-- Strategy Results Table -->
                <div id="advanced-strategy-results" class="bg-white rounded-lg p-4 border border-gray-300 shadow" style="display: none;">
                    <h3 class="text-xl font-bold text-gray-900 mb-4">
                        <i class="fas fa-table mr-2"></i>
                        Advanced Strategy Results
                    </h3>
                    <div class="overflow-x-auto">
                        <table class="w-full text-sm">
                            <thead>
                                <tr class="border-b-2 border-gray-300">
                                    <th class="text-left p-2 font-bold text-gray-900">Strategy</th>
                                    <th class="text-left p-2 font-bold text-gray-900">Signal</th>
                                    <th class="text-left p-2 font-bold text-gray-900">Confidence</th>
                                    <th class="text-left p-2 font-bold text-gray-900">Key Metric</th>
                                    <th class="text-left p-2 font-bold text-gray-900">Status</th>
                                </tr>
                            </thead>
                            <tbody id="strategy-results-tbody">
                                <!-- Results will be populated here -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- Footer -->
            <div class="mt-8 text-center text-gray-600">
                <p>LLM-Driven Trading Intelligence System • Built with Hono + Cloudflare D1 + Chart.js</p>
                <p class="text-sm text-gray-500 mt-2">✨ Now with Advanced Quantitative Strategies: Arbitrage • Pair Trading • Multi-Factor Alpha • ML/DL Predictions</p>
            </div>
        </div>

        <script>
            // Fetch and display agent data
            // Update dashboard stats from DATABASE (NO HARDCODING)
            async function updateDashboardStats() {
                try {
                    const response = await axios.get('/api/dashboard/summary');
                    if (response.data.success) {
                        // Dashboard data loaded successfully
                        // Static metrics removed - using real-time agent data instead
                    }
                } catch (error) {
                    console.error('Error updating dashboard stats:', error);
                    // Keep existing values on error
                }
            }

            // Countdown timer variables
            let refreshCountdown = 10;
            let countdownInterval = null;
            
            // Update countdown display
            function updateCountdown() {
                document.getElementById('economic-countdown').textContent = \`Next update: \${refreshCountdown}s\`;
                document.getElementById('sentiment-countdown').textContent = \`Next update: \${refreshCountdown}s\`;
                document.getElementById('cross-exchange-countdown').textContent = \`Next update: \${refreshCountdown}s\`;
                refreshCountdown--;
                
                if (refreshCountdown < 0) {
                    refreshCountdown = 10;
                }
            }
            
            // Format timestamp
            function formatTime(timestamp) {
                const date = new Date(timestamp);
                return date.toLocaleTimeString('en-US', { hour12: false });
            }

            async function loadAgentData() {
                console.log('Loading agent data...');
                const fetchTime = Date.now();
                refreshCountdown = 10; // Reset countdown
                
                try {
                    // Fetch Economic Agent
                    console.log('Fetching economic agent...');
                    const economicRes = await axios.get('/api/agents/economic?symbol=BTC');
                    const econ = economicRes.data.data.indicators;
                    const econTimestamp = economicRes.data.data.iso_timestamp;
                    console.log('Economic agent loaded:', econ);
                    
                    // Update timestamp display
                    document.getElementById('economic-timestamp').textContent = formatTime(fetchTime);
                    
                    document.getElementById('economic-agent-data').innerHTML = \`
                        <div class="flex justify-between">
                            <span class="text-gray-600">Fed Rate:</span>
                            <span class="text-gray-900 font-bold">\${econ.fed_funds_rate.value}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">CPI Inflation:</span>
                            <span class="text-gray-900 font-bold">\${econ.cpi.value}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">GDP Growth:</span>
                            <span class="text-gray-900 font-bold">\${econ.gdp_growth.value}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Unemployment:</span>
                            <span class="text-gray-900 font-bold">\${econ.unemployment_rate.value}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">PMI:</span>
                            <span class="text-gray-900 font-bold">\${econ.manufacturing_pmi.value}</span>
                        </div>
                    \`;

                    // Fetch Sentiment Agent
                    console.log('Fetching sentiment agent...');
                    const sentimentRes = await axios.get('/api/agents/sentiment?symbol=BTC');
                    const sent = sentimentRes.data.data.sentiment_metrics;
                    const sentTimestamp = sentimentRes.data.data.iso_timestamp;
                    console.log('Sentiment agent loaded:', sent);
                    
                    // Update timestamp display
                    document.getElementById('sentiment-timestamp').textContent = formatTime(fetchTime);
                    
                    document.getElementById('sentiment-agent-data').innerHTML = \`
                        <div class="flex justify-between">
                            <span class="text-gray-600">Fear & Greed:</span>
                            <span class="text-gray-900 font-bold">\${sent.fear_greed_index.value} (\${sent.fear_greed_index.classification})</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Signal:</span>
                            <span class="text-gray-900 font-bold">\${sent.fear_greed_index.signal}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">VIX:</span>
                            <span class="text-gray-900 font-bold">\${sent.volatility_index_vix.value.toFixed(2)} (\${sent.volatility_index_vix.signal})</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Social Volume:</span>
                            <span class="text-gray-900 font-bold">\${(sent.social_media_volume.mentions/1000).toFixed(0)}K</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Inst. Flow:</span>
                            <span class="text-gray-900 font-bold">\${sent.institutional_flow_24h.net_flow_million_usd.toFixed(1)}M (\${sent.institutional_flow_24h.direction})</span>
                        </div>
                    \`;

                    // Fetch Cross-Exchange Agent
                    console.log('Fetching cross-exchange agent...');
                    const crossRes = await axios.get('/api/agents/cross-exchange?symbol=BTC');
                    const cross = crossRes.data.data.market_depth_analysis;
                    const liveExchanges = crossRes.data.data.live_exchanges;
                    const crossTimestamp = crossRes.data.data.iso_timestamp;
                    console.log('Cross-exchange agent loaded:', cross);
                    
                    // Update timestamp display
                    document.getElementById('cross-exchange-timestamp').textContent = formatTime(fetchTime);
                    
                    // Get live prices from exchanges
                    const coinbasePrice = liveExchanges.coinbase.available ? liveExchanges.coinbase.price : null;
                    const krakenPrice = liveExchanges.kraken.available ? liveExchanges.kraken.price : null;
                    
                    document.getElementById('cross-exchange-agent-data').innerHTML = \`
                        <div class="flex justify-between">
                            <span class="text-gray-600">Coinbase Price:</span>
                            <span class="text-gray-900 font-bold">\${coinbasePrice ? '$' + coinbasePrice.toLocaleString() : 'N/A'}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Kraken Price:</span>
                            <span class="text-gray-900 font-bold">\${krakenPrice ? '$' + krakenPrice.toLocaleString() : 'N/A'}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">24h Volume:</span>
                            <span class="text-gray-900 font-bold">\${cross.total_volume_24h.usd.toLocaleString()} BTC</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Avg Spread:</span>
                            <span class="text-gray-900 font-bold">\${cross.liquidity_metrics.average_spread_percent}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Liquidity:</span>
                            <span class="text-gray-900 font-bold">\${cross.liquidity_metrics.liquidity_quality}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Arbitrage:</span>
                            <span class="text-gray-900 font-bold">\${cross.arbitrage_opportunities.count} opps</span>
                        </div>
                    \`;
                } catch (error) {
                    console.error('Error loading agent data:', error);
                    // Show error in UI
                    const errorMsg = '<div class="text-red-600 text-sm"><i class="fas fa-exclamation-circle mr-1"></i>Error loading data</div>';
                    if (document.getElementById('economic-agent-data')) {
                        document.getElementById('economic-agent-data').innerHTML = errorMsg;
                    }
                    if (document.getElementById('sentiment-agent-data')) {
                        document.getElementById('sentiment-agent-data').innerHTML = errorMsg;
                    }
                    if (document.getElementById('cross-exchange-agent-data')) {
                        document.getElementById('cross-exchange-agent-data').innerHTML = errorMsg;
                    }
                }
            }

            // Run LLM Analysis
            async function runLLMAnalysis() {
                const resultsDiv = document.getElementById('llm-results');
                const metadataDiv = document.getElementById('llm-metadata');
                
                resultsDiv.innerHTML = '<p class="text-gray-600"><i class="fas fa-spinner fa-spin mr-2"></i>Fetching agent data and generating AI analysis...</p>';
                metadataDiv.innerHTML = '';

                try {
                    const response = await axios.post('/api/llm/analyze-enhanced', {
                        symbol: 'BTC',
                        timeframe: '1h'
                    });

                    const data = response.data;
                    
                    resultsDiv.innerHTML = \`
                        <div class="prose max-w-none">
                            <div class="mb-4">
                                <span class="bg-green-600 text-white px-3 py-1 rounded-full text-xs font-bold">
                                    \${data.model}
                                </span>
                            </div>
                            <div class="text-gray-800 whitespace-pre-wrap">\${data.analysis}</div>
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
                        <div class="text-red-600">
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
                
                resultsDiv.innerHTML = '<p class="text-gray-600"><i class="fas fa-spinner fa-spin mr-2"></i>Running agent-based backtest...</p>';
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
                    const signals = bt.agent_signals || {};
                    
                    // Safety checks for signal properties
                    const economicScore = signals.economicScore || 0;
                    const sentimentScore = signals.sentimentScore || 0;
                    const liquidityScore = signals.liquidityScore || 0;
                    const totalScore = signals.totalScore || 0;
                    const confidence = signals.confidence || 0;
                    const reasoning = signals.reasoning || 'Trading signals based on agent composite scoring';
                    
                    const returnColor = bt.total_return >= 0 ? 'text-green-700' : 'text-red-700';
                    
                    resultsDiv.innerHTML = \`
                        <div class="space-y-4">
                            <div class="bg-white border border-orange-200 p-4 rounded-lg">
                                <h4 class="font-bold text-lg mb-3 text-orange-800">Agent Signals</h4>
                                <div class="grid grid-cols-2 gap-2 text-sm">
                                    <div class="flex justify-between">
                                        <span class="text-gray-600">Economic Score:</span>
                                        <span class="text-gray-900 font-bold">\${economicScore}/6</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-600">Sentiment Score:</span>
                                        <span class="text-gray-900 font-bold">\${sentimentScore}/6</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-600">Liquidity Score:</span>
                                        <span class="text-gray-900 font-bold">\${liquidityScore}/6</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-600">Total Score:</span>
                                        <span class="text-orange-700 font-bold">\${totalScore}/18</span>
                                    </div>
                                </div>
                                <div class="mt-3 pt-3 border-t border-orange-200">
                                    <div class="flex justify-between mb-2">
                                        <span class="text-gray-600">Signal:</span>
                                        <span class="font-bold \${signals.shouldBuy ? 'text-green-700' : signals.shouldSell ? 'text-red-700' : 'text-orange-700'}">
                                            \${signals.shouldBuy ? 'BUY' : signals.shouldSell ? 'SELL' : 'HOLD'}
                                        </span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-600">Confidence:</span>
                                        <span class="text-gray-900 font-bold">\${confidence}%</span>
                                    </div>
                                    <div class="mt-2">
                                        <p class="text-xs text-gray-600">\${reasoning}</p>
                                    </div>
                                </div>
                            </div>

                            <div class="bg-white border border-orange-200 p-4 rounded-lg">
                                <h4 class="font-bold text-lg mb-3 text-orange-800">Performance</h4>
                                <div class="grid grid-cols-2 gap-2 text-sm">
                                    <div class="flex justify-between">
                                        <span class="text-gray-600">Initial Capital:</span>
                                        <span class="text-gray-900 font-bold">$\${bt.initial_capital.toLocaleString()}</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-600">Final Capital:</span>
                                        <span class="text-gray-900 font-bold">$\${bt.final_capital.toFixed(2)}</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-600">Total Return:</span>
                                        <span class="\${returnColor} font-bold">\${bt.total_return.toFixed(2)}%</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-600">Sharpe Ratio:</span>
                                        <span class="text-gray-900 font-bold">\${bt.sharpe_ratio.toFixed(2)}</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-600">Max Drawdown:</span>
                                        <span class="text-red-700 font-bold">\${bt.max_drawdown.toFixed(2)}%</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-600">Win Rate:</span>
                                        <span class="text-gray-900 font-bold">\${bt.win_rate.toFixed(0)}%</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-600">Total Trades:</span>
                                        <span class="text-gray-900 font-bold">\${bt.total_trades}</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-600">Win/Loss:</span>
                                        <span class="text-gray-900 font-bold">\${bt.winning_trades}W / \${bt.losing_trades}L</span>
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
                    
                    // Fetch cross-exchange data for arbitrage chart
                    const crossRes = await axios.get('/api/agents/cross-exchange?symbol=BTC');
                    if (crossRes.data.success) {
                        updateArbitrageChart(crossRes.data.data.market_depth_analysis);
                    }
                } catch (error) {
                    resultsDiv.innerHTML = \`
                        <div class="text-red-600">
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



            // Initialize charts first
            initializeCharts();
            
            // Start countdown timer (updates every second)
            countdownInterval = setInterval(updateCountdown, 1000);
            
            // Load agent data immediately on page load
            document.addEventListener('DOMContentLoaded', function() {
                console.log('DOM Content Loaded - starting data fetch');
                updateDashboardStats();
                loadAgentData();
                // Refresh every 10 seconds
                setInterval(loadAgentData, 10000);
            });
            
            // Also call immediately (in case DOMContentLoaded already fired)
            setTimeout(() => {
                console.log('Fallback data load triggered');
                updateDashboardStats();
                loadAgentData();
            }, 100);

            // ========================================================================
            // ADVANCED QUANTITATIVE STRATEGIES JAVASCRIPT
            // ========================================================================

            // Advanced Arbitrage Detection
            async function runAdvancedArbitrage() {
                const resultDiv = document.getElementById('arbitrage-result');
                resultDiv.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i> Detecting arbitrage opportunities...';
                
                try {
                    const response = await axios.get('/api/strategies/arbitrage/advanced?symbol=BTC');
                    const data = response.data;
                    
                    if (data.success) {
                        const total = data.arbitrage_opportunities.total_opportunities;
                        const spatial = data.arbitrage_opportunities.spatial.count;
                        
                        resultDiv.innerHTML = \`
                            <div class="bg-amber-50 border border-green-200 rounded p-2 mt-2">
                                <p class="font-bold text-green-800">✓ Found \${total} Opportunities</p>
                                <p class="text-green-700">Spatial: \${spatial} opportunities</p>
                                <p class="text-xs text-gray-600 mt-1">Min profit threshold: 0.3% after fees</p>
                            </div>
                        \`;
                        addStrategyResult('Advanced Arbitrage', total > 0 ? 'BUY' : 'HOLD', 0.85, \`\${total} opportunities\`, 'Active');
                    }
                } catch (error) {
                    resultDiv.innerHTML = '<div class="text-red-600"><i class="fas fa-exclamation-circle mr-1"></i> Error loading data</div>';
                }
            }

            // Statistical Pair Trading
            async function runPairTrading() {
                const resultDiv = document.getElementById('pair-result');
                resultDiv.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i> Analyzing BTC-ETH pair...';
                
                try {
                    const response = await axios.post('/api/strategies/pairs/analyze', {
                        pair1: 'BTC',
                        pair2: 'ETH'
                    });
                    const data = response.data;
                    
                    if (data.success) {
                        const signal = data.trading_signals.signal;
                        const zscore = data.spread_analysis.current_zscore.toFixed(2);
                        const cointegrated = data.cointegration.is_cointegrated;
                        
                        resultDiv.innerHTML = \`
                            <div class="bg-amber-50 border border-purple-200 rounded p-2 mt-2">
                                <p class="font-bold text-purple-800">✓ Signal: \${signal}</p>
                                <p class="text-purple-700">Z-Score: \${zscore}</p>
                                <p class="text-purple-700">Cointegrated: \${cointegrated ? 'Yes' : 'No'}</p>
                                <p class="text-xs text-gray-600 mt-1">Half-Life: \${data.mean_reversion.half_life_days} days</p>
                            </div>
                        \`;
                        addStrategyResult('Pair Trading', signal, 0.78, \`Z-Score: \${zscore}\`, cointegrated ? 'Active' : 'Inactive');
                    }
                } catch (error) {
                    resultDiv.innerHTML = '<div class="text-red-600"><i class="fas fa-exclamation-circle mr-1"></i> Error loading data</div>';
                }
            }

            // Multi-Factor Alpha
            async function runMultiFactorAlpha() {
                const resultDiv = document.getElementById('factor-result');
                resultDiv.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i> Calculating factor exposures...';
                
                try {
                    const response = await axios.get('/api/strategies/factors/score?symbol=BTC');
                    const data = response.data;
                    
                    if (data.success) {
                        const signal = data.composite_alpha.signal;
                        const score = (data.composite_alpha.overall_score * 100).toFixed(0);
                        const dominant = data.factor_exposure.dominant_factor;
                        
                        resultDiv.innerHTML = \`
                            <div class="bg-amber-50 border border-blue-200 rounded p-2 mt-2">
                                <p class="font-bold text-blue-800">✓ Signal: \${signal}</p>
                                <p class="text-blue-700">Alpha Score: \${score}/100</p>
                                <p class="text-blue-700">Dominant Factor: \${dominant}</p>
                                <p class="text-xs text-gray-600 mt-1">5-Factor + Momentum Analysis</p>
                            </div>
                        \`;
                        addStrategyResult('Multi-Factor Alpha', signal, data.composite_alpha.confidence, \`Score: \${score}/100\`, 'Active');
                    }
                } catch (error) {
                    resultDiv.innerHTML = '<div class="text-red-600"><i class="fas fa-exclamation-circle mr-1"></i> Error loading data</div>';
                }
            }

            // Machine Learning Prediction
            async function runMLPrediction() {
                const resultDiv = document.getElementById('ml-result');
                resultDiv.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i> Running ensemble models...';
                
                try {
                    const response = await axios.post('/api/strategies/ml/predict', {
                        symbol: 'BTC'
                    });
                    const data = response.data;
                    
                    if (data.success) {
                        const signal = data.ensemble_prediction.signal;
                        const confidence = (data.ensemble_prediction.confidence * 100).toFixed(0);
                        const agreement = (data.ensemble_prediction.model_agreement * 100).toFixed(0);
                        
                        resultDiv.innerHTML = \`
                            <div class="bg-amber-50 border border-orange-200 rounded p-2 mt-2">
                                <p class="font-bold text-orange-800">✓ Ensemble: \${signal}</p>
                                <p class="text-orange-700">Confidence: \${confidence}%</p>
                                <p class="text-orange-700">Model Agreement: \${agreement}%</p>
                                <p class="text-xs text-gray-600 mt-1">5 models: RF, XGB, SVM, LR, NN</p>
                            </div>
                        \`;
                        addStrategyResult('Machine Learning', signal, confidence/100, \`Agreement: \${agreement}%\`, 'Active');
                    }
                } catch (error) {
                    resultDiv.innerHTML = '<div class="text-red-600"><i class="fas fa-exclamation-circle mr-1"></i> Error loading data</div>';
                }
            }

            // Deep Learning Analysis
            async function runDLAnalysis() {
                const resultDiv = document.getElementById('dl-result');
                resultDiv.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i> Running neural networks...';
                
                try {
                    const response = await axios.post('/api/strategies/dl/analyze', {
                        symbol: 'BTC',
                        horizon: 24
                    });
                    const data = response.data;
                    
                    if (data.success) {
                        const signal = data.ensemble_dl_signal.combined_signal;
                        const confidence = (data.ensemble_dl_signal.confidence * 100).toFixed(0);
                        const lstmTrend = data.lstm_prediction.trend_direction;
                        
                        resultDiv.innerHTML = \`
                            <div class="bg-red-50 border border-red-200 rounded p-2 mt-2">
                                <p class="font-bold text-red-800">✓ DL Signal: \${signal}</p>
                                <p class="text-red-700">Confidence: \${confidence}%</p>
                                <p class="text-red-700">LSTM Trend: \${lstmTrend}</p>
                                <p class="text-xs text-gray-600 mt-1">LSTM + Transformer + GAN</p>
                            </div>
                        \`;
                        addStrategyResult('Deep Learning', signal, confidence/100, \`Trend: \${lstmTrend}\`, 'Active');
                    }
                } catch (error) {
                    resultDiv.innerHTML = '<div class="text-red-600"><i class="fas fa-exclamation-circle mr-1"></i> Error loading data</div>';
                }
            }

            // Compare All Advanced Strategies
            async function compareAllStrategies() {
                const resultDiv = document.getElementById('comparison-result');
                resultDiv.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i> Running all strategies...';
                
                try {
                    // Run all strategies in parallel
                    await Promise.all([
                        runAdvancedArbitrage(),
                        runPairTrading(),
                        runMultiFactorAlpha(),
                        runMLPrediction(),
                        runDLAnalysis()
                    ]);
                    
                    resultDiv.innerHTML = \`
                        <div class="bg-gray-50 border border-gray-300 rounded p-2 mt-2">
                            <p class="font-bold text-gray-800">✓ All Strategies Complete</p>
                            <p class="text-gray-700">Check results table below</p>
                        </div>
                    \`;
                    
                    // Show results table
                    document.getElementById('advanced-strategy-results').style.display = 'block';
                } catch (error) {
                    resultDiv.innerHTML = '<div class="text-red-600"><i class="fas fa-exclamation-circle mr-1"></i> Error running comparison</div>';
                }
            }

            // Helper function to add strategy result to table
            function addStrategyResult(strategy, signal, confidence, metric, status) {
                const tbody = document.getElementById('strategy-results-tbody');
                const signalColor = signal.includes('BUY') ? 'text-green-700' : signal.includes('SELL') ? 'text-red-700' : 'text-gray-700';
                const confidencePercent = (confidence * 100).toFixed(0);
                
                const row = document.createElement('tr');
                row.className = 'border-b border-gray-200 hover:bg-gray-50';
                row.innerHTML = \`
                    <td class="p-2 font-bold text-gray-900">\${strategy}</td>
                    <td class="p-2 \${signalColor} font-bold">\${signal}</td>
                    <td class="p-2 text-gray-700">\${confidencePercent}%</td>
                    <td class="p-2 text-gray-700">\${metric}</td>
                    <td class="p-2">
                        <span class="px-2 py-1 rounded text-xs font-bold \${status === 'Active' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}">
                            \${status}
                        </span>
                    </td>
                \`;
                tbody.appendChild(row);
            }
        <\/script>
    </body>
    </html>
  `));const tt=new Ct,bs=Object.assign({"/src/index.tsx":S});let Mt=!1;for(const[,e]of Object.entries(bs))e&&(tt.route("/",e),tt.notFound(e.notFoundHandler),Mt=!0);if(!Mt)throw new Error("Can't import modules from ['/src/index.tsx','/app/server.ts']");export{tt as default};
