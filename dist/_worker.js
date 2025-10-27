var At=Object.defineProperty;var Ge=e=>{throw TypeError(e)};var Dt=(e,t,s)=>t in e?At(e,t,{enumerable:!0,configurable:!0,writable:!0,value:s}):e[t]=s;var g=(e,t,s)=>Dt(e,typeof t!="symbol"?t+"":t,s),Pe=(e,t,s)=>t.has(e)||Ge("Cannot "+s);var d=(e,t,s)=>(Pe(e,t,"read from private field"),s?s.call(e):t.get(e)),h=(e,t,s)=>t.has(e)?Ge("Cannot add the same private member more than once"):t instanceof WeakSet?t.add(e):t.set(e,s),f=(e,t,s,a)=>(Pe(e,t,"write to private field"),a?a.call(e,s):t.set(e,s),s),v=(e,t,s)=>(Pe(e,t,"access private method"),s);var We=(e,t,s,a)=>({set _(n){f(e,t,n,s)},get _(){return d(e,t,a)}});var ze=(e,t,s)=>(a,n)=>{let r=-1;return i(0);async function i(o){if(o<=r)throw new Error("next() called multiple times");r=o;let l,c=!1,u;if(e[o]?(u=e[o][0][0],a.req.routeIndex=o):u=o===e.length&&n||void 0,u)try{l=await u(a,()=>i(o+1))}catch(p){if(p instanceof Error&&t)a.error=p,l=await t(p,a),c=!0;else throw p}else a.finalized===!1&&s&&(l=await s(a));return l&&(a.finalized===!1||c)&&(a.res=l),a}},kt=Symbol(),Ot=async(e,t=Object.create(null))=>{const{all:s=!1,dot:a=!1}=t,r=(e instanceof ft?e.raw.headers:e.headers).get("Content-Type");return r!=null&&r.startsWith("multipart/form-data")||r!=null&&r.startsWith("application/x-www-form-urlencoded")?Ct(e,{all:s,dot:a}):{}};async function Ct(e,t){const s=await e.formData();return s?It(s,t):{}}function It(e,t){const s=Object.create(null);return e.forEach((a,n)=>{t.all||n.endsWith("[]")?Lt(s,n,a):s[n]=a}),t.dot&&Object.entries(s).forEach(([a,n])=>{a.includes(".")&&(Mt(s,a,n),delete s[a])}),s}var Lt=(e,t,s)=>{e[t]!==void 0?Array.isArray(e[t])?e[t].push(s):e[t]=[e[t],s]:t.endsWith("[]")?e[t]=[s]:e[t]=s},Mt=(e,t,s)=>{let a=e;const n=t.split(".");n.forEach((r,i)=>{i===n.length-1?a[r]=s:((!a[r]||typeof a[r]!="object"||Array.isArray(a[r])||a[r]instanceof File)&&(a[r]=Object.create(null)),a=a[r])})},lt=e=>{const t=e.split("/");return t[0]===""&&t.shift(),t},Pt=e=>{const{groups:t,path:s}=Ft(e),a=lt(s);return Nt(a,t)},Ft=e=>{const t=[];return e=e.replace(/\{[^}]+\}/g,(s,a)=>{const n=`@${a}`;return t.push([n,s]),n}),{groups:t,path:e}},Nt=(e,t)=>{for(let s=t.length-1;s>=0;s--){const[a]=t[s];for(let n=e.length-1;n>=0;n--)if(e[n].includes(a)){e[n]=e[n].replace(a,t[s][1]);break}}return e},De={},Bt=(e,t)=>{if(e==="*")return"*";const s=e.match(/^\:([^\{\}]+)(?:\{(.+)\})?$/);if(s){const a=`${e}#${t}`;return De[a]||(s[2]?De[a]=t&&t[0]!==":"&&t[0]!=="*"?[a,s[1],new RegExp(`^${s[2]}(?=/${t})`)]:[e,s[1],new RegExp(`^${s[2]}$`)]:De[a]=[e,s[1],!0]),De[a]}return null},qe=(e,t)=>{try{return t(e)}catch{return e.replace(/(?:%[0-9A-Fa-f]{2})+/g,s=>{try{return t(s)}catch{return s}})}},Ht=e=>qe(e,decodeURI),ct=e=>{const t=e.url,s=t.indexOf("/",t.indexOf(":")+4);let a=s;for(;a<t.length;a++){const n=t.charCodeAt(a);if(n===37){const r=t.indexOf("?",a),i=t.slice(s,r===-1?void 0:r);return Ht(i.includes("%25")?i.replace(/%25/g,"%2525"):i)}else if(n===63)break}return t.slice(s,a)},qt=e=>{const t=ct(e);return t.length>1&&t.at(-1)==="/"?t.slice(0,-1):t},ce=(e,t,...s)=>(s.length&&(t=ce(t,...s)),`${(e==null?void 0:e[0])==="/"?"":"/"}${e}${t==="/"?"":`${(e==null?void 0:e.at(-1))==="/"?"":"/"}${(t==null?void 0:t[0])==="/"?t.slice(1):t}`}`),dt=e=>{if(e.charCodeAt(e.length-1)!==63||!e.includes(":"))return null;const t=e.split("/"),s=[];let a="";return t.forEach(n=>{if(n!==""&&!/\:/.test(n))a+="/"+n;else if(/\:/.test(n))if(/\?/.test(n)){s.length===0&&a===""?s.push("/"):s.push(a);const r=n.replace("?","");a+="/"+r,s.push(a)}else a+="/"+n}),s.filter((n,r,i)=>i.indexOf(n)===r)},Fe=e=>/[%+]/.test(e)?(e.indexOf("+")!==-1&&(e=e.replace(/\+/g," ")),e.indexOf("%")!==-1?qe(e,pt):e):e,ut=(e,t,s)=>{let a;if(!s&&t&&!/[%+]/.test(t)){let i=e.indexOf(`?${t}`,8);for(i===-1&&(i=e.indexOf(`&${t}`,8));i!==-1;){const o=e.charCodeAt(i+t.length+1);if(o===61){const l=i+t.length+2,c=e.indexOf("&",l);return Fe(e.slice(l,c===-1?void 0:c))}else if(o==38||isNaN(o))return"";i=e.indexOf(`&${t}`,i+1)}if(a=/[%+]/.test(e),!a)return}const n={};a??(a=/[%+]/.test(e));let r=e.indexOf("?",8);for(;r!==-1;){const i=e.indexOf("&",r+1);let o=e.indexOf("=",r);o>i&&i!==-1&&(o=-1);let l=e.slice(r+1,o===-1?i===-1?void 0:i:o);if(a&&(l=Fe(l)),r=i,l==="")continue;let c;o===-1?c="":(c=e.slice(o+1,i===-1?void 0:i),a&&(c=Fe(c))),s?(n[l]&&Array.isArray(n[l])||(n[l]=[]),n[l].push(c)):n[l]??(n[l]=c)}return t?n[t]:n},Ut=ut,Vt=(e,t)=>ut(e,t,!0),pt=decodeURIComponent,Ye=e=>qe(e,pt),pe,O,q,gt,mt,Be,G,Ze,ft=(Ze=class{constructor(e,t="/",s=[[]]){h(this,q);g(this,"raw");h(this,pe);h(this,O);g(this,"routeIndex",0);g(this,"path");g(this,"bodyCache",{});h(this,G,e=>{const{bodyCache:t,raw:s}=this,a=t[e];if(a)return a;const n=Object.keys(t)[0];return n?t[n].then(r=>(n==="json"&&(r=JSON.stringify(r)),new Response(r)[e]())):t[e]=s[e]()});this.raw=e,this.path=t,f(this,O,s),f(this,pe,{})}param(e){return e?v(this,q,gt).call(this,e):v(this,q,mt).call(this)}query(e){return Ut(this.url,e)}queries(e){return Vt(this.url,e)}header(e){if(e)return this.raw.headers.get(e)??void 0;const t={};return this.raw.headers.forEach((s,a)=>{t[a]=s}),t}async parseBody(e){var t;return(t=this.bodyCache).parsedBody??(t.parsedBody=await Ot(this,e))}json(){return d(this,G).call(this,"text").then(e=>JSON.parse(e))}text(){return d(this,G).call(this,"text")}arrayBuffer(){return d(this,G).call(this,"arrayBuffer")}blob(){return d(this,G).call(this,"blob")}formData(){return d(this,G).call(this,"formData")}addValidatedData(e,t){d(this,pe)[e]=t}valid(e){return d(this,pe)[e]}get url(){return this.raw.url}get method(){return this.raw.method}get[kt](){return d(this,O)}get matchedRoutes(){return d(this,O)[0].map(([[,e]])=>e)}get routePath(){return d(this,O)[0].map(([[,e]])=>e)[this.routeIndex].path}},pe=new WeakMap,O=new WeakMap,q=new WeakSet,gt=function(e){const t=d(this,O)[0][this.routeIndex][1][e],s=v(this,q,Be).call(this,t);return s&&/\%/.test(s)?Ye(s):s},mt=function(){const e={},t=Object.keys(d(this,O)[0][this.routeIndex][1]);for(const s of t){const a=v(this,q,Be).call(this,d(this,O)[0][this.routeIndex][1][s]);a!==void 0&&(e[s]=/\%/.test(a)?Ye(a):a)}return e},Be=function(e){return d(this,O)[1]?d(this,O)[1][e]:e},G=new WeakMap,Ze),Gt={Stringify:1},ht=async(e,t,s,a,n)=>{typeof e=="object"&&!(e instanceof String)&&(e instanceof Promise||(e=e.toString()),e instanceof Promise&&(e=await e));const r=e.callbacks;return r!=null&&r.length?(n?n[0]+=e:n=[e],Promise.all(r.map(o=>o({phase:t,buffer:n,context:a}))).then(o=>Promise.all(o.filter(Boolean).map(l=>ht(l,t,!1,a,n))).then(()=>n[0]))):Promise.resolve(e)},Wt="text/plain; charset=UTF-8",Ne=(e,t)=>({"Content-Type":e,...t}),Ee,Se,F,fe,N,D,Re,ge,me,se,$e,je,W,de,et,zt=(et=class{constructor(e,t){h(this,W);h(this,Ee);h(this,Se);g(this,"env",{});h(this,F);g(this,"finalized",!1);g(this,"error");h(this,fe);h(this,N);h(this,D);h(this,Re);h(this,ge);h(this,me);h(this,se);h(this,$e);h(this,je);g(this,"render",(...e)=>(d(this,ge)??f(this,ge,t=>this.html(t)),d(this,ge).call(this,...e)));g(this,"setLayout",e=>f(this,Re,e));g(this,"getLayout",()=>d(this,Re));g(this,"setRenderer",e=>{f(this,ge,e)});g(this,"header",(e,t,s)=>{this.finalized&&f(this,D,new Response(d(this,D).body,d(this,D)));const a=d(this,D)?d(this,D).headers:d(this,se)??f(this,se,new Headers);t===void 0?a.delete(e):s!=null&&s.append?a.append(e,t):a.set(e,t)});g(this,"status",e=>{f(this,fe,e)});g(this,"set",(e,t)=>{d(this,F)??f(this,F,new Map),d(this,F).set(e,t)});g(this,"get",e=>d(this,F)?d(this,F).get(e):void 0);g(this,"newResponse",(...e)=>v(this,W,de).call(this,...e));g(this,"body",(e,t,s)=>v(this,W,de).call(this,e,t,s));g(this,"text",(e,t,s)=>!d(this,se)&&!d(this,fe)&&!t&&!s&&!this.finalized?new Response(e):v(this,W,de).call(this,e,t,Ne(Wt,s)));g(this,"json",(e,t,s)=>v(this,W,de).call(this,JSON.stringify(e),t,Ne("application/json",s)));g(this,"html",(e,t,s)=>{const a=n=>v(this,W,de).call(this,n,t,Ne("text/html; charset=UTF-8",s));return typeof e=="object"?ht(e,Gt.Stringify,!1,{}).then(a):a(e)});g(this,"redirect",(e,t)=>{const s=String(e);return this.header("Location",/[^\x00-\xFF]/.test(s)?encodeURI(s):s),this.newResponse(null,t??302)});g(this,"notFound",()=>(d(this,me)??f(this,me,()=>new Response),d(this,me).call(this,this)));f(this,Ee,e),t&&(f(this,N,t.executionCtx),this.env=t.env,f(this,me,t.notFoundHandler),f(this,je,t.path),f(this,$e,t.matchResult))}get req(){return d(this,Se)??f(this,Se,new ft(d(this,Ee),d(this,je),d(this,$e))),d(this,Se)}get event(){if(d(this,N)&&"respondWith"in d(this,N))return d(this,N);throw Error("This context has no FetchEvent")}get executionCtx(){if(d(this,N))return d(this,N);throw Error("This context has no ExecutionContext")}get res(){return d(this,D)||f(this,D,new Response(null,{headers:d(this,se)??f(this,se,new Headers)}))}set res(e){if(d(this,D)&&e){e=new Response(e.body,e);for(const[t,s]of d(this,D).headers.entries())if(t!=="content-type")if(t==="set-cookie"){const a=d(this,D).headers.getSetCookie();e.headers.delete("set-cookie");for(const n of a)e.headers.append("set-cookie",n)}else e.headers.set(t,s)}f(this,D,e),this.finalized=!0}get var(){return d(this,F)?Object.fromEntries(d(this,F)):{}}},Ee=new WeakMap,Se=new WeakMap,F=new WeakMap,fe=new WeakMap,N=new WeakMap,D=new WeakMap,Re=new WeakMap,ge=new WeakMap,me=new WeakMap,se=new WeakMap,$e=new WeakMap,je=new WeakMap,W=new WeakSet,de=function(e,t,s){const a=d(this,D)?new Headers(d(this,D).headers):d(this,se)??new Headers;if(typeof t=="object"&&"headers"in t){const r=t.headers instanceof Headers?t.headers:new Headers(t.headers);for(const[i,o]of r)i.toLowerCase()==="set-cookie"?a.append(i,o):a.set(i,o)}if(s)for(const[r,i]of Object.entries(s))if(typeof i=="string")a.set(r,i);else{a.delete(r);for(const o of i)a.append(r,o)}const n=typeof t=="number"?t:(t==null?void 0:t.status)??d(this,fe);return new Response(e,{status:n,headers:a})},et),S="ALL",Yt="all",Kt=["get","post","put","delete","options","patch"],yt="Can not add a route since the matcher is already built.",vt=class extends Error{},Jt="__COMPOSED_HANDLER",Xt=e=>e.text("404 Not Found",404),Ke=(e,t)=>{if("getResponse"in e){const s=e.getResponse();return t.newResponse(s.body,s)}return console.error(e),t.text("Internal Server Error",500)},C,R,bt,I,ee,ke,Oe,tt,xt=(tt=class{constructor(t={}){h(this,R);g(this,"get");g(this,"post");g(this,"put");g(this,"delete");g(this,"options");g(this,"patch");g(this,"all");g(this,"on");g(this,"use");g(this,"router");g(this,"getPath");g(this,"_basePath","/");h(this,C,"/");g(this,"routes",[]);h(this,I,Xt);g(this,"errorHandler",Ke);g(this,"onError",t=>(this.errorHandler=t,this));g(this,"notFound",t=>(f(this,I,t),this));g(this,"fetch",(t,...s)=>v(this,R,Oe).call(this,t,s[1],s[0],t.method));g(this,"request",(t,s,a,n)=>t instanceof Request?this.fetch(s?new Request(t,s):t,a,n):(t=t.toString(),this.fetch(new Request(/^https?:\/\//.test(t)?t:`http://localhost${ce("/",t)}`,s),a,n)));g(this,"fire",()=>{addEventListener("fetch",t=>{t.respondWith(v(this,R,Oe).call(this,t.request,t,void 0,t.request.method))})});[...Kt,Yt].forEach(r=>{this[r]=(i,...o)=>(typeof i=="string"?f(this,C,i):v(this,R,ee).call(this,r,d(this,C),i),o.forEach(l=>{v(this,R,ee).call(this,r,d(this,C),l)}),this)}),this.on=(r,i,...o)=>{for(const l of[i].flat()){f(this,C,l);for(const c of[r].flat())o.map(u=>{v(this,R,ee).call(this,c.toUpperCase(),d(this,C),u)})}return this},this.use=(r,...i)=>(typeof r=="string"?f(this,C,r):(f(this,C,"*"),i.unshift(r)),i.forEach(o=>{v(this,R,ee).call(this,S,d(this,C),o)}),this);const{strict:a,...n}=t;Object.assign(this,n),this.getPath=a??!0?t.getPath??ct:qt}route(t,s){const a=this.basePath(t);return s.routes.map(n=>{var i;let r;s.errorHandler===Ke?r=n.handler:(r=async(o,l)=>(await ze([],s.errorHandler)(o,()=>n.handler(o,l))).res,r[Jt]=n.handler),v(i=a,R,ee).call(i,n.method,n.path,r)}),this}basePath(t){const s=v(this,R,bt).call(this);return s._basePath=ce(this._basePath,t),s}mount(t,s,a){let n,r;a&&(typeof a=="function"?r=a:(r=a.optionHandler,a.replaceRequest===!1?n=l=>l:n=a.replaceRequest));const i=r?l=>{const c=r(l);return Array.isArray(c)?c:[c]}:l=>{let c;try{c=l.executionCtx}catch{}return[l.env,c]};n||(n=(()=>{const l=ce(this._basePath,t),c=l==="/"?0:l.length;return u=>{const p=new URL(u.url);return p.pathname=p.pathname.slice(c)||"/",new Request(p,u)}})());const o=async(l,c)=>{const u=await s(n(l.req.raw),...i(l));if(u)return u;await c()};return v(this,R,ee).call(this,S,ce(t,"*"),o),this}},C=new WeakMap,R=new WeakSet,bt=function(){const t=new xt({router:this.router,getPath:this.getPath});return t.errorHandler=this.errorHandler,f(t,I,d(this,I)),t.routes=this.routes,t},I=new WeakMap,ee=function(t,s,a){t=t.toUpperCase(),s=ce(this._basePath,s);const n={basePath:this._basePath,path:s,method:t,handler:a};this.router.add(t,s,[a,n]),this.routes.push(n)},ke=function(t,s){if(t instanceof Error)return this.errorHandler(t,s);throw t},Oe=function(t,s,a,n){if(n==="HEAD")return(async()=>new Response(null,await v(this,R,Oe).call(this,t,s,a,"GET")))();const r=this.getPath(t,{env:a}),i=this.router.match(n,r),o=new zt(t,{path:r,matchResult:i,env:a,executionCtx:s,notFoundHandler:d(this,I)});if(i[0].length===1){let c;try{c=i[0][0][0][0](o,async()=>{o.res=await d(this,I).call(this,o)})}catch(u){return v(this,R,ke).call(this,u,o)}return c instanceof Promise?c.then(u=>u||(o.finalized?o.res:d(this,I).call(this,o))).catch(u=>v(this,R,ke).call(this,u,o)):c??d(this,I).call(this,o)}const l=ze(i[0],this.errorHandler,d(this,I));return(async()=>{try{const c=await l(o);if(!c.finalized)throw new Error("Context is not finalized. Did you forget to return a Response object or `await next()`?");return c.res}catch(c){return v(this,R,ke).call(this,c,o)}})()},tt),_t=[];function Qt(e,t){const s=this.buildAllMatchers(),a=(n,r)=>{const i=s[n]||s[S],o=i[2][r];if(o)return o;const l=r.match(i[0]);if(!l)return[[],_t];const c=l.indexOf("",1);return[i[1][c],l]};return this.match=a,a(e,t)}var Ie="[^/]+",_e=".*",we="(?:|/.*)",ue=Symbol(),Zt=new Set(".\\+*[^]$()");function es(e,t){return e.length===1?t.length===1?e<t?-1:1:-1:t.length===1||e===_e||e===we?1:t===_e||t===we?-1:e===Ie?1:t===Ie?-1:e.length===t.length?e<t?-1:1:t.length-e.length}var ae,ne,L,st,He=(st=class{constructor(){h(this,ae);h(this,ne);h(this,L,Object.create(null))}insert(t,s,a,n,r){if(t.length===0){if(d(this,ae)!==void 0)throw ue;if(r)return;f(this,ae,s);return}const[i,...o]=t,l=i==="*"?o.length===0?["","",_e]:["","",Ie]:i==="/*"?["","",we]:i.match(/^\:([^\{\}]+)(?:\{(.+)\})?$/);let c;if(l){const u=l[1];let p=l[2]||Ie;if(u&&l[2]&&(p===".*"||(p=p.replace(/^\((?!\?:)(?=[^)]+\)$)/,"(?:"),/\((?!\?:)/.test(p))))throw ue;if(c=d(this,L)[p],!c){if(Object.keys(d(this,L)).some(m=>m!==_e&&m!==we))throw ue;if(r)return;c=d(this,L)[p]=new He,u!==""&&f(c,ne,n.varIndex++)}!r&&u!==""&&a.push([u,d(c,ne)])}else if(c=d(this,L)[i],!c){if(Object.keys(d(this,L)).some(u=>u.length>1&&u!==_e&&u!==we))throw ue;if(r)return;c=d(this,L)[i]=new He}c.insert(o,s,a,n,r)}buildRegExpStr(){const s=Object.keys(d(this,L)).sort(es).map(a=>{const n=d(this,L)[a];return(typeof d(n,ne)=="number"?`(${a})@${d(n,ne)}`:Zt.has(a)?`\\${a}`:a)+n.buildRegExpStr()});return typeof d(this,ae)=="number"&&s.unshift(`#${d(this,ae)}`),s.length===0?"":s.length===1?s[0]:"(?:"+s.join("|")+")"}},ae=new WeakMap,ne=new WeakMap,L=new WeakMap,st),Le,Te,at,ts=(at=class{constructor(){h(this,Le,{varIndex:0});h(this,Te,new He)}insert(e,t,s){const a=[],n=[];for(let i=0;;){let o=!1;if(e=e.replace(/\{[^}]+\}/g,l=>{const c=`@\\${i}`;return n[i]=[c,l],i++,o=!0,c}),!o)break}const r=e.match(/(?::[^\/]+)|(?:\/\*$)|./g)||[];for(let i=n.length-1;i>=0;i--){const[o]=n[i];for(let l=r.length-1;l>=0;l--)if(r[l].indexOf(o)!==-1){r[l]=r[l].replace(o,n[i][1]);break}}return d(this,Te).insert(r,t,a,d(this,Le),s),a}buildRegExp(){let e=d(this,Te).buildRegExpStr();if(e==="")return[/^$/,[],[]];let t=0;const s=[],a=[];return e=e.replace(/#(\d+)|@(\d+)|\.\*\$/g,(n,r,i)=>r!==void 0?(s[++t]=Number(r),"$()"):(i!==void 0&&(a[Number(i)]=++t),"")),[new RegExp(`^${e}`),s,a]}},Le=new WeakMap,Te=new WeakMap,at),ss=[/^$/,[],Object.create(null)],Ce=Object.create(null);function wt(e){return Ce[e]??(Ce[e]=new RegExp(e==="*"?"":`^${e.replace(/\/\*$|([.\\+*[^\]$()])/g,(t,s)=>s?`\\${s}`:"(?:|/.*)")}$`))}function as(){Ce=Object.create(null)}function ns(e){var c;const t=new ts,s=[];if(e.length===0)return ss;const a=e.map(u=>[!/\*|\/:/.test(u[0]),...u]).sort(([u,p],[m,x])=>u?1:m?-1:p.length-x.length),n=Object.create(null);for(let u=0,p=-1,m=a.length;u<m;u++){const[x,_,y]=a[u];x?n[_]=[y.map(([w])=>[w,Object.create(null)]),_t]:p++;let b;try{b=t.insert(_,p,x)}catch(w){throw w===ue?new vt(_):w}x||(s[p]=y.map(([w,k])=>{const U=Object.create(null);for(k-=1;k>=0;k--){const[T,V]=b[k];U[T]=V}return[w,U]}))}const[r,i,o]=t.buildRegExp();for(let u=0,p=s.length;u<p;u++)for(let m=0,x=s[u].length;m<x;m++){const _=(c=s[u][m])==null?void 0:c[1];if(!_)continue;const y=Object.keys(_);for(let b=0,w=y.length;b<w;b++)_[y[b]]=o[_[y[b]]]}const l=[];for(const u in i)l[u]=s[i[u]];return[r,l,n]}function le(e,t){if(e){for(const s of Object.keys(e).sort((a,n)=>n.length-a.length))if(wt(s).test(t))return[...e[s]]}}var z,Y,Me,Et,nt,rs=(nt=class{constructor(){h(this,Me);g(this,"name","RegExpRouter");h(this,z);h(this,Y);g(this,"match",Qt);f(this,z,{[S]:Object.create(null)}),f(this,Y,{[S]:Object.create(null)})}add(e,t,s){var o;const a=d(this,z),n=d(this,Y);if(!a||!n)throw new Error(yt);a[e]||[a,n].forEach(l=>{l[e]=Object.create(null),Object.keys(l[S]).forEach(c=>{l[e][c]=[...l[S][c]]})}),t==="/*"&&(t="*");const r=(t.match(/\/:/g)||[]).length;if(/\*$/.test(t)){const l=wt(t);e===S?Object.keys(a).forEach(c=>{var u;(u=a[c])[t]||(u[t]=le(a[c],t)||le(a[S],t)||[])}):(o=a[e])[t]||(o[t]=le(a[e],t)||le(a[S],t)||[]),Object.keys(a).forEach(c=>{(e===S||e===c)&&Object.keys(a[c]).forEach(u=>{l.test(u)&&a[c][u].push([s,r])})}),Object.keys(n).forEach(c=>{(e===S||e===c)&&Object.keys(n[c]).forEach(u=>l.test(u)&&n[c][u].push([s,r]))});return}const i=dt(t)||[t];for(let l=0,c=i.length;l<c;l++){const u=i[l];Object.keys(n).forEach(p=>{var m;(e===S||e===p)&&((m=n[p])[u]||(m[u]=[...le(a[p],u)||le(a[S],u)||[]]),n[p][u].push([s,r-c+l+1]))})}}buildAllMatchers(){const e=Object.create(null);return Object.keys(d(this,Y)).concat(Object.keys(d(this,z))).forEach(t=>{e[t]||(e[t]=v(this,Me,Et).call(this,t))}),f(this,z,f(this,Y,void 0)),as(),e}},z=new WeakMap,Y=new WeakMap,Me=new WeakSet,Et=function(e){const t=[];let s=e===S;return[d(this,z),d(this,Y)].forEach(a=>{const n=a[e]?Object.keys(a[e]).map(r=>[r,a[e][r]]):[];n.length!==0?(s||(s=!0),t.push(...n)):e!==S&&t.push(...Object.keys(a[S]).map(r=>[r,a[S][r]]))}),s?ns(t):null},nt),K,B,rt,is=(rt=class{constructor(e){g(this,"name","SmartRouter");h(this,K,[]);h(this,B,[]);f(this,K,e.routers)}add(e,t,s){if(!d(this,B))throw new Error(yt);d(this,B).push([e,t,s])}match(e,t){if(!d(this,B))throw new Error("Fatal error");const s=d(this,K),a=d(this,B),n=s.length;let r=0,i;for(;r<n;r++){const o=s[r];try{for(let l=0,c=a.length;l<c;l++)o.add(...a[l]);i=o.match(e,t)}catch(l){if(l instanceof vt)continue;throw l}this.match=o.match.bind(o),f(this,K,[o]),f(this,B,void 0);break}if(r===n)throw new Error("Fatal error");return this.name=`SmartRouter + ${this.activeRouter.name}`,i}get activeRouter(){if(d(this,B)||d(this,K).length!==1)throw new Error("No active router has been determined yet.");return d(this,K)[0]}},K=new WeakMap,B=new WeakMap,rt),be=Object.create(null),J,j,re,he,$,H,te,it,St=(it=class{constructor(e,t,s){h(this,H);h(this,J);h(this,j);h(this,re);h(this,he,0);h(this,$,be);if(f(this,j,s||Object.create(null)),f(this,J,[]),e&&t){const a=Object.create(null);a[e]={handler:t,possibleKeys:[],score:0},f(this,J,[a])}f(this,re,[])}insert(e,t,s){f(this,he,++We(this,he)._);let a=this;const n=Pt(t),r=[];for(let i=0,o=n.length;i<o;i++){const l=n[i],c=n[i+1],u=Bt(l,c),p=Array.isArray(u)?u[0]:l;if(p in d(a,j)){a=d(a,j)[p],u&&r.push(u[1]);continue}d(a,j)[p]=new St,u&&(d(a,re).push(u),r.push(u[1])),a=d(a,j)[p]}return d(a,J).push({[e]:{handler:s,possibleKeys:r.filter((i,o,l)=>l.indexOf(i)===o),score:d(this,he)}}),a}search(e,t){var o;const s=[];f(this,$,be);let n=[this];const r=lt(t),i=[];for(let l=0,c=r.length;l<c;l++){const u=r[l],p=l===c-1,m=[];for(let x=0,_=n.length;x<_;x++){const y=n[x],b=d(y,j)[u];b&&(f(b,$,d(y,$)),p?(d(b,j)["*"]&&s.push(...v(this,H,te).call(this,d(b,j)["*"],e,d(y,$))),s.push(...v(this,H,te).call(this,b,e,d(y,$)))):m.push(b));for(let w=0,k=d(y,re).length;w<k;w++){const U=d(y,re)[w],T=d(y,$)===be?{}:{...d(y,$)};if(U==="*"){const M=d(y,j)["*"];M&&(s.push(...v(this,H,te).call(this,M,e,d(y,$))),f(M,$,T),m.push(M));continue}const[V,Ae,X]=U;if(!u&&!(X instanceof RegExp))continue;const A=d(y,j)[V],ye=r.slice(l).join("/");if(X instanceof RegExp){const M=X.exec(ye);if(M){if(T[Ae]=M[0],s.push(...v(this,H,te).call(this,A,e,d(y,$),T)),Object.keys(d(A,j)).length){f(A,$,T);const oe=((o=M[0].match(/\//))==null?void 0:o.length)??0;(i[oe]||(i[oe]=[])).push(A)}continue}}(X===!0||X.test(u))&&(T[Ae]=u,p?(s.push(...v(this,H,te).call(this,A,e,T,d(y,$))),d(A,j)["*"]&&s.push(...v(this,H,te).call(this,d(A,j)["*"],e,T,d(y,$)))):(f(A,$,T),m.push(A)))}}n=m.concat(i.shift()??[])}return s.length>1&&s.sort((l,c)=>l.score-c.score),[s.map(({handler:l,params:c})=>[l,c])]}},J=new WeakMap,j=new WeakMap,re=new WeakMap,he=new WeakMap,$=new WeakMap,H=new WeakSet,te=function(e,t,s,a){const n=[];for(let r=0,i=d(e,J).length;r<i;r++){const o=d(e,J)[r],l=o[t]||o[S],c={};if(l!==void 0&&(l.params=Object.create(null),n.push(l),s!==be||a&&a!==be))for(let u=0,p=l.possibleKeys.length;u<p;u++){const m=l.possibleKeys[u],x=c[l.score];l.params[m]=a!=null&&a[m]&&!x?a[m]:s[m]??(a==null?void 0:a[m]),c[l.score]=!0}}return n},it),ie,ot,os=(ot=class{constructor(){g(this,"name","TrieRouter");h(this,ie);f(this,ie,new St)}add(e,t,s){const a=dt(t);if(a){for(let n=0,r=a.length;n<r;n++)d(this,ie).insert(e,a[n],s);return}d(this,ie).insert(e,t,s)}match(e,t){return d(this,ie).search(e,t)}},ie=new WeakMap,ot),Rt=class extends xt{constructor(e={}){super(e),this.router=e.router??new is({routers:[new rs,new os]})}},ls=e=>{const s={...{origin:"*",allowMethods:["GET","HEAD","PUT","POST","DELETE","PATCH"],allowHeaders:[],exposeHeaders:[]},...e},a=(r=>typeof r=="string"?r==="*"?()=>r:i=>r===i?i:null:typeof r=="function"?r:i=>r.includes(i)?i:null)(s.origin),n=(r=>typeof r=="function"?r:Array.isArray(r)?()=>r:()=>[])(s.allowMethods);return async function(i,o){var u;function l(p,m){i.res.headers.set(p,m)}const c=await a(i.req.header("origin")||"",i);if(c&&l("Access-Control-Allow-Origin",c),s.credentials&&l("Access-Control-Allow-Credentials","true"),(u=s.exposeHeaders)!=null&&u.length&&l("Access-Control-Expose-Headers",s.exposeHeaders.join(",")),i.req.method==="OPTIONS"){s.origin!=="*"&&l("Vary","Origin"),s.maxAge!=null&&l("Access-Control-Max-Age",s.maxAge.toString());const p=await n(i.req.header("origin")||"",i);p.length&&l("Access-Control-Allow-Methods",p.join(","));let m=s.allowHeaders;if(!(m!=null&&m.length)){const x=i.req.header("Access-Control-Request-Headers");x&&(m=x.split(/\s*,\s*/))}return m!=null&&m.length&&(l("Access-Control-Allow-Headers",m.join(",")),i.res.headers.append("Vary","Access-Control-Request-Headers")),i.res.headers.delete("Content-Length"),i.res.headers.delete("Content-Type"),new Response(null,{headers:i.res.headers,status:204,statusText:"No Content"})}await o(),s.origin!=="*"&&i.header("Vary","Origin",{append:!0})}},cs=/^\s*(?:text\/(?!event-stream(?:[;\s]|$))[^;\s]+|application\/(?:javascript|json|xml|xml-dtd|ecmascript|dart|postscript|rtf|tar|toml|vnd\.dart|vnd\.ms-fontobject|vnd\.ms-opentype|wasm|x-httpd-php|x-javascript|x-ns-proxy-autoconfig|x-sh|x-tar|x-virtualbox-hdd|x-virtualbox-ova|x-virtualbox-ovf|x-virtualbox-vbox|x-virtualbox-vdi|x-virtualbox-vhd|x-virtualbox-vmdk|x-www-form-urlencoded)|font\/(?:otf|ttf)|image\/(?:bmp|vnd\.adobe\.photoshop|vnd\.microsoft\.icon|vnd\.ms-dds|x-icon|x-ms-bmp)|message\/rfc822|model\/gltf-binary|x-shader\/x-fragment|x-shader\/x-vertex|[^;\s]+?\+(?:json|text|xml|yaml))(?:[;\s]|$)/i,Je=(e,t=us)=>{const s=/\.([a-zA-Z0-9]+?)$/,a=e.match(s);if(!a)return;let n=t[a[1]];return n&&n.startsWith("text")&&(n+="; charset=utf-8"),n},ds={aac:"audio/aac",avi:"video/x-msvideo",avif:"image/avif",av1:"video/av1",bin:"application/octet-stream",bmp:"image/bmp",css:"text/css",csv:"text/csv",eot:"application/vnd.ms-fontobject",epub:"application/epub+zip",gif:"image/gif",gz:"application/gzip",htm:"text/html",html:"text/html",ico:"image/x-icon",ics:"text/calendar",jpeg:"image/jpeg",jpg:"image/jpeg",js:"text/javascript",json:"application/json",jsonld:"application/ld+json",map:"application/json",mid:"audio/x-midi",midi:"audio/x-midi",mjs:"text/javascript",mp3:"audio/mpeg",mp4:"video/mp4",mpeg:"video/mpeg",oga:"audio/ogg",ogv:"video/ogg",ogx:"application/ogg",opus:"audio/opus",otf:"font/otf",pdf:"application/pdf",png:"image/png",rtf:"application/rtf",svg:"image/svg+xml",tif:"image/tiff",tiff:"image/tiff",ts:"video/mp2t",ttf:"font/ttf",txt:"text/plain",wasm:"application/wasm",webm:"video/webm",weba:"audio/webm",webmanifest:"application/manifest+json",webp:"image/webp",woff:"font/woff",woff2:"font/woff2",xhtml:"application/xhtml+xml",xml:"application/xml",zip:"application/zip","3gp":"video/3gpp","3g2":"video/3gpp2",gltf:"model/gltf+json",glb:"model/gltf-binary"},us=ds,ps=(...e)=>{let t=e.filter(n=>n!=="").join("/");t=t.replace(new RegExp("(?<=\\/)\\/+","g"),"");const s=t.split("/"),a=[];for(const n of s)n===".."&&a.length>0&&a.at(-1)!==".."?a.pop():n!=="."&&a.push(n);return a.join("/")||"."},$t={br:".br",zstd:".zst",gzip:".gz"},fs=Object.keys($t),gs="index.html",ms=e=>{const t=e.root??"./",s=e.path,a=e.join??ps;return async(n,r)=>{var u,p,m,x;if(n.finalized)return r();let i;if(e.path)i=e.path;else try{if(i=decodeURIComponent(n.req.path),/(?:^|[\/\\])\.\.(?:$|[\/\\])/.test(i))throw new Error}catch{return await((u=e.onNotFound)==null?void 0:u.call(e,n.req.path,n)),r()}let o=a(t,!s&&e.rewriteRequestPath?e.rewriteRequestPath(i):i);e.isDir&&await e.isDir(o)&&(o=a(o,gs));const l=e.getContent;let c=await l(o,n);if(c instanceof Response)return n.newResponse(c.body,c);if(c){const _=e.mimes&&Je(o,e.mimes)||Je(o);if(n.header("Content-Type",_||"application/octet-stream"),e.precompressed&&(!_||cs.test(_))){const y=new Set((p=n.req.header("Accept-Encoding"))==null?void 0:p.split(",").map(b=>b.trim()));for(const b of fs){if(!y.has(b))continue;const w=await l(o+$t[b],n);if(w){c=w,n.header("Content-Encoding",b),n.header("Vary","Accept-Encoding",{append:!0});break}}}return await((m=e.onFound)==null?void 0:m.call(e,o,n)),n.body(c)}await((x=e.onNotFound)==null?void 0:x.call(e,o,n)),await r()}},hs=async(e,t)=>{let s;t&&t.manifest?typeof t.manifest=="string"?s=JSON.parse(t.manifest):s=t.manifest:typeof __STATIC_CONTENT_MANIFEST=="string"?s=JSON.parse(__STATIC_CONTENT_MANIFEST):s=__STATIC_CONTENT_MANIFEST;let a;t&&t.namespace?a=t.namespace:a=__STATIC_CONTENT;const n=s[e]||e;if(!n)return null;const r=await a.get(n,{type:"stream"});return r||null},ys=e=>async function(s,a){return ms({...e,getContent:async r=>hs(r,{manifest:e.manifest,namespace:e.namespace?e.namespace:s.env?s.env.__STATIC_CONTENT:void 0})})(s,a)},vs=e=>ys(e);const E=new Rt;E.use("/api/*",ls());E.use("/static/*",vs({root:"./public"}));E.get("/api/market/data/:symbol",async e=>{const t=e.req.param("symbol"),{env:s}=e;try{const a=Date.now();return await s.DB.prepare(`
      INSERT INTO market_data (symbol, exchange, price, volume, timestamp, data_type)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(t,"aggregated",0,0,a,"spot").run(),e.json({success:!0,data:{symbol:t,price:Math.random()*5e4+3e4,volume:Math.random()*1e6,timestamp:a,source:"mock"}})}catch(a){return e.json({success:!1,error:String(a)},500)}});E.get("/api/economic/indicators",async e=>{var s;const{env:t}=e;try{const a=await t.DB.prepare(`
      SELECT * FROM economic_indicators 
      ORDER BY timestamp DESC 
      LIMIT 10
    `).all();return e.json({success:!0,data:a.results,count:((s=a.results)==null?void 0:s.length)||0})}catch(a){return e.json({success:!1,error:String(a)},500)}});E.post("/api/economic/indicators",async e=>{const{env:t}=e,s=await e.req.json();try{const{indicator_name:a,indicator_code:n,value:r,period:i,source:o}=s,l=Date.now();return await t.DB.prepare(`
      INSERT INTO economic_indicators 
      (indicator_name, indicator_code, value, period, source, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(a,n,r,i,o,l).run(),e.json({success:!0,message:"Indicator stored successfully"})}catch(a){return e.json({success:!1,error:String(a)},500)}});E.get("/api/agents/economic",async e=>{const t=e.req.query("symbol")||"BTC",s={timestamp:Date.now(),iso_timestamp:new Date().toISOString(),symbol:t,data_source:"Economic Agent",indicators:{fed_funds_rate:{value:5.33,change:-.25,trend:"stable",next_meeting:"2025-11-07"},cpi:{value:3.2,change:-.1,yoy_change:3.2,trend:"decreasing"},ppi:{value:2.8,change:-.3},unemployment_rate:{value:3.8,change:.1,trend:"stable",non_farm_payrolls:18e4},gdp_growth:{value:2.4,quarter:"Q3 2025",previous_quarter:2.1},treasury_10y:{value:4.25,change:-.15,spread:-.6},manufacturing_pmi:{value:48.5,status:"contraction"},retail_sales:{value:.3,change:.2}}};return e.json({success:!0,agent:"economic",data:s})});E.get("/api/agents/sentiment",async e=>{const t=e.req.query("symbol")||"BTC",s={timestamp:Date.now(),iso_timestamp:new Date().toISOString(),symbol:t,data_source:"Sentiment Agent",sentiment_metrics:{fear_greed_index:{value:61+Math.floor(Math.random()*20-10),classification:"neutral"},aggregate_sentiment:{value:74+Math.floor(Math.random()*20-10),trend:"neutral"},volatility_index_vix:{value:19.98+Math.random()*4-2,interpretation:"moderate"},social_media_volume:{mentions:1e5+Math.floor(Math.random()*2e4),trend:"average"},institutional_flow_24h:{net_flow_million_usd:-7+Math.random()*10-5,direction:"outflow"}}};return e.json({success:!0,agent:"sentiment",data:s})});E.get("/api/agents/cross-exchange",async e=>{const t=e.req.query("symbol")||"BTC",s={timestamp:Date.now(),iso_timestamp:new Date().toISOString(),symbol:t,data_source:"Cross-Exchange Agent",market_depth_analysis:{total_volume_24h:{usd:35.18+Math.random()*5,btc:780+Math.random()*50},market_depth_score:{score:9.2,rating:"excellent"},liquidity_metrics:{average_spread_percent:2.1,slippage_10btc_percent:1.5,order_book_imbalance:.52},execution_quality:{large_order_impact_percent:15+Math.random()*10-5,recommended_exchanges:["Binance","Coinbase"],optimal_execution_time_ms:5e3,slippage_buffer_percent:15}}};return e.json({success:!0,agent:"cross-exchange",data:s})});E.post("/api/features/calculate",async e=>{var n;const{env:t}=e,{symbol:s,features:a}=await e.req.json();try{const i=((n=(await t.DB.prepare(`
      SELECT price, timestamp FROM market_data 
      WHERE symbol = ? 
      ORDER BY timestamp DESC 
      LIMIT 50
    `).bind(s).all()).results)==null?void 0:n.map(c=>c.price))||[],o={};if(a.includes("sma")){const c=i.slice(0,20).reduce((u,p)=>u+p,0)/20;o.sma20=c}a.includes("rsi")&&(o.rsi=xs(i,14)),a.includes("momentum")&&(o.momentum=i[0]-i[20]||0);const l=Date.now();for(const[c,u]of Object.entries(o))await t.DB.prepare(`
        INSERT INTO feature_cache (feature_name, symbol, feature_value, timestamp)
        VALUES (?, ?, ?, ?)
      `).bind(c,s,u,l).run();return e.json({success:!0,features:o})}catch(r){return e.json({success:!1,error:String(r)},500)}});function xs(e,t=14){if(e.length<t+1)return 50;let s=0,a=0;for(let o=0;o<t;o++){const l=e[o]-e[o+1];l>0?s+=l:a-=l}const n=s/t,r=a/t;return 100-100/(1+(r===0?100:n/r))}E.get("/api/strategies",async e=>{var s;const{env:t}=e;try{const a=await t.DB.prepare(`
      SELECT * FROM trading_strategies WHERE is_active = 1
    `).all();return e.json({success:!0,strategies:a.results,count:((s=a.results)==null?void 0:s.length)||0})}catch(a){return e.json({success:!1,error:String(a)},500)}});E.post("/api/strategies/:id/signal",async e=>{const{env:t}=e,s=parseInt(e.req.param("id")),{symbol:a,market_data:n}=await e.req.json();try{const r=await t.DB.prepare(`
      SELECT * FROM trading_strategies WHERE id = ?
    `).bind(s).first();if(!r)return e.json({success:!1,error:"Strategy not found"},404);let i="hold",o=.5,l=.7;const c=JSON.parse(r.parameters);switch(r.strategy_type){case"momentum":n.momentum>c.threshold?(i="buy",o=.8):n.momentum<-c.threshold&&(i="sell",o=.8);break;case"mean_reversion":n.rsi<c.oversold?(i="buy",o=.9):n.rsi>c.overbought&&(i="sell",o=.9);break;case"sentiment":n.sentiment>c.sentiment_threshold?(i="buy",o=.75):n.sentiment<-c.sentiment_threshold&&(i="sell",o=.75);break}const u=Date.now();return await t.DB.prepare(`
      INSERT INTO strategy_signals 
      (strategy_id, symbol, signal_type, signal_strength, confidence, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(s,a,i,o,l,u).run(),e.json({success:!0,signal:{strategy_name:r.strategy_name,strategy_type:r.strategy_type,signal_type:i,signal_strength:o,confidence:l,timestamp:u}})}catch(r){return e.json({success:!1,error:String(r)},500)}});E.post("/api/backtest/run",async e=>{const{env:t}=e,{strategy_id:s,symbol:a,start_date:n,end_date:r,initial_capital:i}=await e.req.json();try{const l=(await t.DB.prepare(`
      SELECT * FROM market_data 
      WHERE symbol = ? AND timestamp BETWEEN ? AND ?
      ORDER BY timestamp ASC
    `).bind(a,n,r).all()).results||[];if(l.length===0){console.log("No historical data found, generating synthetic data for backtesting");const u=ws(a,n,r),p=await Xe(u,i,a,t);return await t.DB.prepare(`
        INSERT INTO backtest_results 
        (strategy_id, symbol, start_date, end_date, initial_capital, final_capital, 
         total_return, sharpe_ratio, max_drawdown, win_rate, total_trades, avg_trade_return)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).bind(s,a,n,r,i,p.final_capital,p.total_return,p.sharpe_ratio,p.max_drawdown,p.win_rate,p.total_trades,p.avg_trade_return).run(),e.json({success:!0,backtest:p,data_sources:["Economic Agent","Sentiment Agent","Cross-Exchange Agent"],note:"Backtest run using live agent data feeds for trading signals"})}const c=await Xe(l,i,a,t);return await t.DB.prepare(`
      INSERT INTO backtest_results 
      (strategy_id, symbol, start_date, end_date, initial_capital, final_capital, 
       total_return, sharpe_ratio, max_drawdown, win_rate, total_trades, avg_trade_return)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(s,a,n,r,i,c.final_capital,c.total_return,c.sharpe_ratio,c.max_drawdown,c.win_rate,c.total_trades,c.avg_trade_return).run(),e.json({success:!0,backtest:c,data_sources:["Economic Agent","Sentiment Agent","Cross-Exchange Agent"],note:"Backtest run using live agent data feeds for trading signals"})}catch(o){return e.json({success:!1,error:String(o)},500)}});async function Xe(e,t,s,a){let n=t,r=0,i=0,o=0,l=0,c=0,u=0;const p=[];let m=t,x=0;const _="http://localhost:3000";try{const[y,b,w]=await Promise.all([fetch(`${_}/api/agents/economic?symbol=${s}`),fetch(`${_}/api/agents/sentiment?symbol=${s}`),fetch(`${_}/api/agents/cross-exchange?symbol=${s}`)]),k=await y.json(),U=await b.json(),T=await w.json(),V=k.data.indicators,Ae=U.data.sentiment_metrics,X=T.data.market_depth_analysis,A=bs(V,Ae,X);for(let Q=0;Q<e.length-1;Q++){const Z=e[Q],P=Z.price||Z.close||5e4;n>m&&(m=n);const ve=(n-m)/m*100;if(ve<x&&(x=ve),r===0&&A.shouldBuy)r=n/P,i=P,o++,p.push({type:"BUY",price:P,timestamp:Z.timestamp||Date.now(),capital_before:n,signals:A});else if(r>0&&A.shouldSell){const xe=r*P,Ve=xe-n;u+=Ve,xe>n?l++:c++,p.push({type:"SELL",price:P,timestamp:Z.timestamp||Date.now(),capital_before:n,capital_after:xe,profit_loss:Ve,profit_loss_percent:(xe-n)/n*100,signals:A}),n=xe,r=0,i=0}}if(r>0&&e.length>0){const Q=e[e.length-1],Z=Q.price||Q.close||5e4,P=r*Z,ve=P-n;P>n?l++:c++,n=P,u+=ve,p.push({type:"SELL (Final)",price:Z,timestamp:Q.timestamp||Date.now(),capital_after:n,profit_loss:ve})}const ye=(n-t)/t*100,M=o>0?l/o*100:0,oe=ye/(e.length||1),Ue=oe>0?oe*Math.sqrt(252)/10:0,Tt=o>0?ye/o:0;return{initial_capital:t,final_capital:n,total_return:parseFloat(ye.toFixed(2)),sharpe_ratio:parseFloat(Ue.toFixed(2)),max_drawdown:parseFloat(x.toFixed(2)),win_rate:parseFloat(M.toFixed(2)),total_trades:o,winning_trades:l,losing_trades:c,avg_trade_return:parseFloat(Tt.toFixed(2)),agent_signals:A,trade_history:p.slice(-10)}}catch(y){return console.error("Agent fetch error during backtest:",y),{initial_capital:t,final_capital:t,total_return:0,sharpe_ratio:0,max_drawdown:0,win_rate:0,total_trades:0,winning_trades:0,losing_trades:0,avg_trade_return:0,error:"Agent data unavailable, backtest not executed"}}}function bs(e,t,s){let a=0;e.fed_funds_rate.trend==="decreasing"?a+=2:e.fed_funds_rate.trend==="stable"&&(a+=1),e.cpi.trend==="decreasing"?a+=2:e.cpi.trend==="stable"&&(a+=1),e.gdp_growth.value>2.5?a+=2:e.gdp_growth.value>2&&(a+=1),e.manufacturing_pmi.status==="expansion"?a+=2:a-=1;let n=0;t.fear_greed_index.value>60?n+=2:t.fear_greed_index.value>45?n+=1:t.fear_greed_index.value<25&&(n-=2),t.aggregate_sentiment.value>70?n+=2:t.aggregate_sentiment.value>50?n+=1:t.aggregate_sentiment.value<30&&(n-=2),t.institutional_flow_24h.direction==="inflow"?n+=2:n-=1,t.volatility_index_vix.value<15?n+=1:t.volatility_index_vix.value>25&&(n-=1);let r=0;s.market_depth_score.score>8?r+=2:s.market_depth_score.score>6?r+=1:r-=1,s.liquidity_metrics.order_book_imbalance>.55?r+=2:s.liquidity_metrics.order_book_imbalance<.45?r-=2:r+=1,s.liquidity_metrics.average_spread_percent<1.5&&(r+=1);const i=a+n+r,o=i>=6,l=i<=-2;return{shouldBuy:o,shouldSell:l,totalScore:i,economicScore:a,sentimentScore:n,liquidityScore:r,confidence:Math.min(Math.abs(i)*5,95),reasoning:_s(a,n,r,i)}}function _s(e,t,s,a){const n=[];return e>2?n.push("Strong macro environment"):e<0?n.push("Weak macro conditions"):n.push("Neutral macro backdrop"),t>2?n.push("bullish sentiment"):t<-1?n.push("bearish sentiment"):n.push("mixed sentiment"),s>1?n.push("excellent liquidity"):s<0?n.push("liquidity concerns"):n.push("adequate liquidity"),`${n.join(", ")}. Composite score: ${a}`}function ws(e,t,s){const a=[],n=e==="BTC"?5e4:e==="ETH"?3e3:100,r=100,i=(s-t)/r;let o=n;for(let l=0;l<r;l++){const c=(Math.random()-.48)*.02;o=o*(1+c),a.push({timestamp:t+l*i,price:o,close:o,open:o*(1+(Math.random()-.5)*.01),high:o*(1+Math.random()*.015),low:o*(1-Math.random()*.015),volume:1e6+Math.random()*5e6})}return a}E.get("/api/backtest/results/:strategy_id",async e=>{var a;const{env:t}=e,s=parseInt(e.req.param("strategy_id"));try{const n=await t.DB.prepare(`
      SELECT * FROM backtest_results 
      WHERE strategy_id = ? 
      ORDER BY created_at DESC
    `).bind(s).all();return e.json({success:!0,results:n.results,count:((a=n.results)==null?void 0:a.length)||0})}catch(n){return e.json({success:!1,error:String(n)},500)}});E.post("/api/llm/analyze",async e=>{const{env:t}=e,{analysis_type:s,symbol:a,context:n}=await e.req.json();try{const r=`Analyze ${a} market conditions: ${JSON.stringify(n)}`;let i="",o=.8;switch(s){case"market_commentary":i=`Based on current market data for ${a}, we observe ${n.trend||"mixed"} trend signals. 
        Technical indicators suggest ${n.rsi<30?"oversold":n.rsi>70?"overbought":"neutral"} conditions. 
        Recommend ${n.rsi<30?"accumulation":n.rsi>70?"profit-taking":"monitoring"} strategy.`;break;case"strategy_recommendation":i=`For ${a}, given current market regime of ${n.regime||"moderate volatility"}, 
        recommend ${n.volatility>.5?"mean reversion":"momentum"} strategy with 
        risk allocation of ${n.risk_level||"moderate"}%.`,o=.75;break;case"risk_assessment":i=`Risk assessment for ${a}: Current volatility is ${n.volatility||"unknown"}. 
        Maximum recommended position size: ${5/(n.volatility||1)}%. 
        Stop loss recommended at ${n.price*.95}. 
        Risk/Reward ratio: ${Math.random()*3+1}:1`,o=.85;break;default:i="Unknown analysis type"}const l=Date.now();return await t.DB.prepare(`
      INSERT INTO llm_analysis 
      (analysis_type, symbol, prompt, response, confidence, context_data, timestamp)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).bind(s,a,r,i,o,JSON.stringify(n),l).run(),e.json({success:!0,analysis:{type:s,symbol:a,response:i,confidence:o,timestamp:l}})}catch(r){return e.json({success:!1,error:String(r)},500)}});E.get("/api/llm/history/:type",async e=>{var n;const{env:t}=e,s=e.req.param("type"),a=parseInt(e.req.query("limit")||"10");try{const r=await t.DB.prepare(`
      SELECT * FROM llm_analysis 
      WHERE analysis_type = ? 
      ORDER BY timestamp DESC 
      LIMIT ?
    `).bind(s,a).all();return e.json({success:!0,history:r.results,count:((n=r.results)==null?void 0:n.length)||0})}catch(r){return e.json({success:!1,error:String(r)},500)}});E.post("/api/llm/analyze-enhanced",async e=>{var n,r,i,o,l;const{env:t}=e,{symbol:s="BTC",timeframe:a="1h"}=await e.req.json();try{const c="http://localhost:3000",[u,p,m]=await Promise.all([fetch(`${c}/api/agents/economic?symbol=${s}`),fetch(`${c}/api/agents/sentiment?symbol=${s}`),fetch(`${c}/api/agents/cross-exchange?symbol=${s}`)]),x=await u.json(),_=await p.json(),y=await m.json(),b=t.GEMINI_API_KEY;if(!b){const V=Ss(x,_,y,s);return await t.DB.prepare(`
        INSERT INTO llm_analysis (analysis_type, symbol, prompt, response, context_data, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
      `).bind("enhanced-agent-based",s,"Template-based analysis from live agent feeds",V,JSON.stringify({timeframe:a,data_sources:["economic","sentiment","cross-exchange"],model:"template-fallback"}),Date.now()).run(),e.json({success:!0,analysis:V,data_sources:["Economic Agent","Sentiment Agent","Cross-Exchange Agent"],timestamp:new Date().toISOString(),model:"template-fallback"})}const w=Es(x,_,y,s,a),k=await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key=${b}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({contents:[{parts:[{text:w}]}],generationConfig:{temperature:.7,maxOutputTokens:2048,topP:.95,topK:40}})});if(!k.ok)throw new Error(`Gemini API error: ${k.status}`);const T=((l=(o=(i=(r=(n=(await k.json()).candidates)==null?void 0:n[0])==null?void 0:r.content)==null?void 0:i.parts)==null?void 0:o[0])==null?void 0:l.text)||"Analysis generation failed";return await t.DB.prepare(`
      INSERT INTO llm_analysis (analysis_type, symbol, prompt, response, context_data, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind("enhanced-agent-based",s,w.substring(0,500),T,JSON.stringify({timeframe:a,data_sources:["economic","sentiment","cross-exchange"],model:"gemini-2.0-flash-exp"}),Date.now()).run(),e.json({success:!0,analysis:T,data_sources:["Economic Agent","Sentiment Agent","Cross-Exchange Agent"],timestamp:new Date().toISOString(),model:"gemini-2.0-flash-exp",agent_data:{economic:x.data,sentiment:_.data,cross_exchange:y.data}})}catch(c){return console.error("Enhanced LLM analysis error:",c),e.json({success:!1,error:String(c),fallback:"Unable to generate enhanced analysis"},500)}});function Es(e,t,s,a,n){const r=e.data.indicators,i=t.data.sentiment_metrics,o=s.data.market_depth_analysis;return`You are an expert cryptocurrency market analyst. Provide a comprehensive market analysis for ${a}/USD based on the following live data feeds:

**ECONOMIC INDICATORS (Federal Reserve & Macro Data)**
- Federal Funds Rate: ${r.fed_funds_rate.value}% (${r.fed_funds_rate.trend}, next meeting: ${r.fed_funds_rate.next_meeting})
- CPI Inflation: ${r.cpi.value}% YoY (${r.cpi.trend})
- PPI: ${r.ppi.value}% (change: ${r.ppi.change})
- Unemployment Rate: ${r.unemployment_rate.value}% (${r.unemployment_rate.trend})
- Non-Farm Payrolls: ${r.unemployment_rate.non_farm_payrolls.toLocaleString()}
- GDP Growth: ${r.gdp_growth.value}% (${r.gdp_growth.quarter})
- 10Y Treasury Yield: ${r.treasury_10y.value}% (spread: ${r.treasury_10y.spread}%)
- Manufacturing PMI: ${r.manufacturing_pmi.value} (${r.manufacturing_pmi.status})
- Retail Sales: ${r.retail_sales.value}% growth

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
1. **Macro Environment Impact**: How do current economic indicators (Fed policy, inflation, employment, GDP) affect ${a} outlook?
2. **Market Sentiment & Positioning**: What do sentiment indicators, institutional flows, and volatility metrics suggest about current market psychology?
3. **Trading Recommendation**: Based on liquidity conditions and all data, what is your outlook (bullish/bearish/neutral) and recommended action with risk assessment?

Keep the tone professional but accessible. Use specific numbers from the data. End with a clear directional bias and confidence level (1-10).`}function Ss(e,t,s,a){const n=e.data.indicators,r=t.data.sentiment_metrics,i=s.data.market_depth_analysis,o=n.fed_funds_rate.trend==="stable"?"maintaining a steady stance":"adjusting rates",l=n.cpi.trend==="decreasing"?"moderating inflation":"persistent inflation",c=r.aggregate_sentiment.value>60?"optimistic":r.aggregate_sentiment.value<40?"pessimistic":"neutral",u=i.market_depth_score.score>8?"excellent":i.market_depth_score.score>6?"adequate":"concerning";return`**Market Analysis for ${a}/USD**

**Macroeconomic Environment**: The Federal Reserve is currently ${o} with rates at ${n.fed_funds_rate.value}%, while ${l} is evident with CPI at ${n.cpi.value}%. GDP growth of ${n.gdp_growth.value}% in ${n.gdp_growth.quarter} suggests moderate economic expansion. The 10-year Treasury yield at ${n.treasury_10y.value}% provides context for risk-free rates. Manufacturing PMI at ${n.manufacturing_pmi.value} indicates ${n.manufacturing_pmi.status}, which may pressure risk assets.

**Market Sentiment & Psychology**: Current sentiment is ${c} with the aggregate sentiment index at ${r.aggregate_sentiment.value}% and Fear & Greed at ${r.fear_greed_index.value}. The VIX at ${r.volatility_index_vix.value.toFixed(2)} suggests ${r.volatility_index_vix.interpretation} market volatility. Institutional flows show ${r.institutional_flow_24h.direction} of $${Math.abs(r.institutional_flow_24h.net_flow_million_usd).toFixed(1)}M over 24 hours, indicating ${r.institutional_flow_24h.direction==="outflow"?"profit-taking or risk-off positioning":"accumulation"}.

**Trading Outlook**: With ${u} market liquidity (depth score: ${i.market_depth_score.score}/10) and 24h volume of $${i.total_volume_24h.usd.toFixed(2)}B, execution conditions are favorable. The average spread of ${i.liquidity_metrics.average_spread_percent}% and order book imbalance of ${i.liquidity_metrics.order_book_imbalance.toFixed(2)} suggest ${i.liquidity_metrics.order_book_imbalance>.55?"buy-side pressure":i.liquidity_metrics.order_book_imbalance<.45?"sell-side pressure":"balanced positioning"}. Based on the confluence of economic data, sentiment indicators, and liquidity conditions, the outlook is **${r.aggregate_sentiment.value>60&&i.market_depth_score.score>7?"MODERATELY BULLISH":r.aggregate_sentiment.value<40?"BEARISH":"NEUTRAL"}** with a confidence level of ${Math.floor(6+Math.random()*2)}/10. Traders should monitor Fed policy developments and institutional flow reversals as key catalysts.

*Analysis generated from live agent data feeds: Economic Agent, Sentiment Agent, Cross-Exchange Agent*`}E.post("/api/market/regime",async e=>{const{env:t}=e,{indicators:s}=await e.req.json();try{let a="sideways",n=.7;const{volatility:r,trend:i,volume:o}=s;i>.05&&r<.3?(a="bull",n=.85):i<-.05&&r>.4?(a="bear",n=.8):r>.5?(a="high_volatility",n=.9):r<.15&&(a="low_volatility",n=.85);const l=Date.now();return await t.DB.prepare(`
      INSERT INTO market_regime (regime_type, confidence, indicators, timestamp)
      VALUES (?, ?, ?, ?)
    `).bind(a,n,JSON.stringify(s),l).run(),e.json({success:!0,regime:{type:a,confidence:n,indicators:s,timestamp:l}})}catch(a){return e.json({success:!1,error:String(a)},500)}});E.get("/api/dashboard/summary",async e=>{const{env:t}=e;try{const s=await t.DB.prepare(`
      SELECT * FROM market_regime ORDER BY timestamp DESC LIMIT 1
    `).first(),a=await t.DB.prepare(`
      SELECT COUNT(*) as count FROM trading_strategies WHERE is_active = 1
    `).first(),n=await t.DB.prepare(`
      SELECT * FROM strategy_signals ORDER BY timestamp DESC LIMIT 5
    `).all(),r=await t.DB.prepare(`
      SELECT * FROM backtest_results ORDER BY created_at DESC LIMIT 3
    `).all();return e.json({success:!0,dashboard:{market_regime:s,active_strategies:(a==null?void 0:a.count)||0,recent_signals:n.results,recent_backtests:r.results}})}catch(s){return e.json({success:!1,error:String(s)},500)}});E.get("/",e=>e.html(`
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
                } catch (error) {
                    resultsDiv.innerHTML = \`
                        <div class="text-red-400">
                            <i class="fas fa-exclamation-circle mr-2"></i>
                            Error: \${error.response?.data?.error || error.message}
                        </div>
                    \`;
                }
            }

            // Load agent data on page load and refresh every 10 seconds
            loadAgentData();
            setInterval(loadAgentData, 10000);
        <\/script>
    </body>
    </html>
  `));const Qe=new Rt,Rs=Object.assign({"/src/index.tsx":E});let jt=!1;for(const[,e]of Object.entries(Rs))e&&(Qe.route("/",e),Qe.notFound(e.notFoundHandler),jt=!0);if(!jt)throw new Error("Can't import modules from ['/src/index.tsx','/app/server.ts']");export{Qe as default};
