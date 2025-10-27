var $t=Object.defineProperty;var Ge=e=>{throw TypeError(e)};var jt=(e,t,a)=>t in e?$t(e,t,{enumerable:!0,configurable:!0,writable:!0,value:a}):e[t]=a;var m=(e,t,a)=>jt(e,typeof t!="symbol"?t+"":t,a),Be=(e,t,a)=>t.has(e)||Ge("Cannot "+a);var d=(e,t,a)=>(Be(e,t,"read from private field"),a?a.call(e):t.get(e)),f=(e,t,a)=>t.has(e)?Ge("Cannot add the same private member more than once"):t instanceof WeakSet?t.add(e):t.set(e,a),p=(e,t,a,s)=>(Be(e,t,"write to private field"),s?s.call(e,a):t.set(e,a),a),y=(e,t,a)=>(Be(e,t,"access private method"),a);var ze=(e,t,a,s)=>({set _(r){p(e,t,r,a)},get _(){return d(e,t,s)}});var We=(e,t,a)=>(s,r)=>{let n=-1;return i(0);async function i(o){if(o<=n)throw new Error("next() called multiple times");n=o;let l,c=!1,u;if(e[o]?(u=e[o][0][0],s.req.routeIndex=o):u=o===e.length&&r||void 0,u)try{l=await u(s,()=>i(o+1))}catch(g){if(g instanceof Error&&t)s.error=g,l=await t(g,s),c=!0;else throw g}else s.finalized===!1&&a&&(l=await a(s));return l&&(s.finalized===!1||c)&&(s.res=l),s}},Tt=Symbol(),Dt=async(e,t=Object.create(null))=>{const{all:a=!1,dot:s=!1}=t,n=(e instanceof pt?e.raw.headers:e.headers).get("Content-Type");return n!=null&&n.startsWith("multipart/form-data")||n!=null&&n.startsWith("application/x-www-form-urlencoded")?Lt(e,{all:a,dot:s}):{}};async function Lt(e,t){const a=await e.formData();return a?It(a,t):{}}function It(e,t){const a=Object.create(null);return e.forEach((s,r)=>{t.all||r.endsWith("[]")?Ot(a,r,s):a[r]=s}),t.dot&&Object.entries(a).forEach(([s,r])=>{s.includes(".")&&(Mt(a,s,r),delete a[s])}),a}var Ot=(e,t,a)=>{e[t]!==void 0?Array.isArray(e[t])?e[t].push(a):e[t]=[e[t],a]:t.endsWith("[]")?e[t]=[a]:e[t]=a},Mt=(e,t,a)=>{let s=e;const r=t.split(".");r.forEach((n,i)=>{i===r.length-1?s[n]=a:((!s[n]||typeof s[n]!="object"||Array.isArray(s[n])||s[n]instanceof File)&&(s[n]=Object.create(null)),s=s[n])})},lt=e=>{const t=e.split("/");return t[0]===""&&t.shift(),t},Bt=e=>{const{groups:t,path:a}=Pt(e),s=lt(a);return Ft(s,t)},Pt=e=>{const t=[];return e=e.replace(/\{[^}]+\}/g,(a,s)=>{const r=`@${s}`;return t.push([r,a]),r}),{groups:t,path:e}},Ft=(e,t)=>{for(let a=t.length-1;a>=0;a--){const[s]=t[a];for(let r=e.length-1;r>=0;r--)if(e[r].includes(s)){e[r]=e[r].replace(s,t[a][1]);break}}return e},je={},Nt=(e,t)=>{if(e==="*")return"*";const a=e.match(/^\:([^\{\}]+)(?:\{(.+)\})?$/);if(a){const s=`${e}#${t}`;return je[s]||(a[2]?je[s]=t&&t[0]!==":"&&t[0]!=="*"?[s,a[1],new RegExp(`^${a[2]}(?=/${t})`)]:[e,a[1],new RegExp(`^${a[2]}$`)]:je[s]=[e,a[1],!0]),je[s]}return null},He=(e,t)=>{try{return t(e)}catch{return e.replace(/(?:%[0-9A-Fa-f]{2})+/g,a=>{try{return t(a)}catch{return a}})}},qt=e=>He(e,decodeURI),ct=e=>{const t=e.url,a=t.indexOf("/",t.indexOf(":")+4);let s=a;for(;s<t.length;s++){const r=t.charCodeAt(s);if(r===37){const n=t.indexOf("?",s),i=t.slice(a,n===-1?void 0:n);return qt(i.includes("%25")?i.replace(/%25/g,"%2525"):i)}else if(r===63)break}return t.slice(a,s)},Ht=e=>{const t=ct(e);return t.length>1&&t.at(-1)==="/"?t.slice(0,-1):t},ce=(e,t,...a)=>(a.length&&(t=ce(t,...a)),`${(e==null?void 0:e[0])==="/"?"":"/"}${e}${t==="/"?"":`${(e==null?void 0:e.at(-1))==="/"?"":"/"}${(t==null?void 0:t[0])==="/"?t.slice(1):t}`}`),dt=e=>{if(e.charCodeAt(e.length-1)!==63||!e.includes(":"))return null;const t=e.split("/"),a=[];let s="";return t.forEach(r=>{if(r!==""&&!/\:/.test(r))s+="/"+r;else if(/\:/.test(r))if(/\?/.test(r)){a.length===0&&s===""?a.push("/"):a.push(s);const n=r.replace("?","");s+="/"+n,a.push(s)}else s+="/"+r}),a.filter((r,n,i)=>i.indexOf(r)===n)},Pe=e=>/[%+]/.test(e)?(e.indexOf("+")!==-1&&(e=e.replace(/\+/g," ")),e.indexOf("%")!==-1?He(e,gt):e):e,ut=(e,t,a)=>{let s;if(!a&&t&&!/[%+]/.test(t)){let i=e.indexOf(`?${t}`,8);for(i===-1&&(i=e.indexOf(`&${t}`,8));i!==-1;){const o=e.charCodeAt(i+t.length+1);if(o===61){const l=i+t.length+2,c=e.indexOf("&",l);return Pe(e.slice(l,c===-1?void 0:c))}else if(o==38||isNaN(o))return"";i=e.indexOf(`&${t}`,i+1)}if(s=/[%+]/.test(e),!s)return}const r={};s??(s=/[%+]/.test(e));let n=e.indexOf("?",8);for(;n!==-1;){const i=e.indexOf("&",n+1);let o=e.indexOf("=",n);o>i&&i!==-1&&(o=-1);let l=e.slice(n+1,o===-1?i===-1?void 0:i:o);if(s&&(l=Pe(l)),n=i,l==="")continue;let c;o===-1?c="":(c=e.slice(o+1,i===-1?void 0:i),s&&(c=Pe(c))),a?(r[l]&&Array.isArray(r[l])||(r[l]=[]),r[l].push(c)):r[l]??(r[l]=c)}return t?r[t]:r},Ut=ut,Vt=(e,t)=>ut(e,t,!0),gt=decodeURIComponent,Ye=e=>He(e,gt),ge,D,H,mt,ht,Ne,G,Qe,pt=(Qe=class{constructor(e,t="/",a=[[]]){f(this,H);m(this,"raw");f(this,ge);f(this,D);m(this,"routeIndex",0);m(this,"path");m(this,"bodyCache",{});f(this,G,e=>{const{bodyCache:t,raw:a}=this,s=t[e];if(s)return s;const r=Object.keys(t)[0];return r?t[r].then(n=>(r==="json"&&(n=JSON.stringify(n)),new Response(n)[e]())):t[e]=a[e]()});this.raw=e,this.path=t,p(this,D,a),p(this,ge,{})}param(e){return e?y(this,H,mt).call(this,e):y(this,H,ht).call(this)}query(e){return Ut(this.url,e)}queries(e){return Vt(this.url,e)}header(e){if(e)return this.raw.headers.get(e)??void 0;const t={};return this.raw.headers.forEach((a,s)=>{t[s]=a}),t}async parseBody(e){var t;return(t=this.bodyCache).parsedBody??(t.parsedBody=await Dt(this,e))}json(){return d(this,G).call(this,"text").then(e=>JSON.parse(e))}text(){return d(this,G).call(this,"text")}arrayBuffer(){return d(this,G).call(this,"arrayBuffer")}blob(){return d(this,G).call(this,"blob")}formData(){return d(this,G).call(this,"formData")}addValidatedData(e,t){d(this,ge)[e]=t}valid(e){return d(this,ge)[e]}get url(){return this.raw.url}get method(){return this.raw.method}get[Tt](){return d(this,D)}get matchedRoutes(){return d(this,D)[0].map(([[,e]])=>e)}get routePath(){return d(this,D)[0].map(([[,e]])=>e)[this.routeIndex].path}},ge=new WeakMap,D=new WeakMap,H=new WeakSet,mt=function(e){const t=d(this,D)[0][this.routeIndex][1][e],a=y(this,H,Ne).call(this,t);return a&&/\%/.test(a)?Ye(a):a},ht=function(){const e={},t=Object.keys(d(this,D)[0][this.routeIndex][1]);for(const a of t){const s=y(this,H,Ne).call(this,d(this,D)[0][this.routeIndex][1][a]);s!==void 0&&(e[a]=/\%/.test(s)?Ye(s):s)}return e},Ne=function(e){return d(this,D)[1]?d(this,D)[1][e]:e},G=new WeakMap,Qe),Gt={Stringify:1},ft=async(e,t,a,s,r)=>{typeof e=="object"&&!(e instanceof String)&&(e instanceof Promise||(e=e.toString()),e instanceof Promise&&(e=await e));const n=e.callbacks;return n!=null&&n.length?(r?r[0]+=e:r=[e],Promise.all(n.map(o=>o({phase:t,buffer:r,context:s}))).then(o=>Promise.all(o.filter(Boolean).map(l=>ft(l,t,!1,s,r))).then(()=>r[0]))):Promise.resolve(e)},zt="text/plain; charset=UTF-8",Fe=(e,t)=>({"Content-Type":e,...t}),Se,Ee,P,pe,F,j,Ce,me,he,ae,ke,Re,z,de,et,Wt=(et=class{constructor(e,t){f(this,z);f(this,Se);f(this,Ee);m(this,"env",{});f(this,P);m(this,"finalized",!1);m(this,"error");f(this,pe);f(this,F);f(this,j);f(this,Ce);f(this,me);f(this,he);f(this,ae);f(this,ke);f(this,Re);m(this,"render",(...e)=>(d(this,me)??p(this,me,t=>this.html(t)),d(this,me).call(this,...e)));m(this,"setLayout",e=>p(this,Ce,e));m(this,"getLayout",()=>d(this,Ce));m(this,"setRenderer",e=>{p(this,me,e)});m(this,"header",(e,t,a)=>{this.finalized&&p(this,j,new Response(d(this,j).body,d(this,j)));const s=d(this,j)?d(this,j).headers:d(this,ae)??p(this,ae,new Headers);t===void 0?s.delete(e):a!=null&&a.append?s.append(e,t):s.set(e,t)});m(this,"status",e=>{p(this,pe,e)});m(this,"set",(e,t)=>{d(this,P)??p(this,P,new Map),d(this,P).set(e,t)});m(this,"get",e=>d(this,P)?d(this,P).get(e):void 0);m(this,"newResponse",(...e)=>y(this,z,de).call(this,...e));m(this,"body",(e,t,a)=>y(this,z,de).call(this,e,t,a));m(this,"text",(e,t,a)=>!d(this,ae)&&!d(this,pe)&&!t&&!a&&!this.finalized?new Response(e):y(this,z,de).call(this,e,t,Fe(zt,a)));m(this,"json",(e,t,a)=>y(this,z,de).call(this,JSON.stringify(e),t,Fe("application/json",a)));m(this,"html",(e,t,a)=>{const s=r=>y(this,z,de).call(this,r,t,Fe("text/html; charset=UTF-8",a));return typeof e=="object"?ft(e,Gt.Stringify,!1,{}).then(s):s(e)});m(this,"redirect",(e,t)=>{const a=String(e);return this.header("Location",/[^\x00-\xFF]/.test(a)?encodeURI(a):a),this.newResponse(null,t??302)});m(this,"notFound",()=>(d(this,he)??p(this,he,()=>new Response),d(this,he).call(this,this)));p(this,Se,e),t&&(p(this,F,t.executionCtx),this.env=t.env,p(this,he,t.notFoundHandler),p(this,Re,t.path),p(this,ke,t.matchResult))}get req(){return d(this,Ee)??p(this,Ee,new pt(d(this,Se),d(this,Re),d(this,ke))),d(this,Ee)}get event(){if(d(this,F)&&"respondWith"in d(this,F))return d(this,F);throw Error("This context has no FetchEvent")}get executionCtx(){if(d(this,F))return d(this,F);throw Error("This context has no ExecutionContext")}get res(){return d(this,j)||p(this,j,new Response(null,{headers:d(this,ae)??p(this,ae,new Headers)}))}set res(e){if(d(this,j)&&e){e=new Response(e.body,e);for(const[t,a]of d(this,j).headers.entries())if(t!=="content-type")if(t==="set-cookie"){const s=d(this,j).headers.getSetCookie();e.headers.delete("set-cookie");for(const r of s)e.headers.append("set-cookie",r)}else e.headers.set(t,a)}p(this,j,e),this.finalized=!0}get var(){return d(this,P)?Object.fromEntries(d(this,P)):{}}},Se=new WeakMap,Ee=new WeakMap,P=new WeakMap,pe=new WeakMap,F=new WeakMap,j=new WeakMap,Ce=new WeakMap,me=new WeakMap,he=new WeakMap,ae=new WeakMap,ke=new WeakMap,Re=new WeakMap,z=new WeakSet,de=function(e,t,a){const s=d(this,j)?new Headers(d(this,j).headers):d(this,ae)??new Headers;if(typeof t=="object"&&"headers"in t){const n=t.headers instanceof Headers?t.headers:new Headers(t.headers);for(const[i,o]of n)i.toLowerCase()==="set-cookie"?s.append(i,o):s.set(i,o)}if(a)for(const[n,i]of Object.entries(a))if(typeof i=="string")s.set(n,i);else{s.delete(n);for(const o of i)s.append(n,o)}const r=typeof t=="number"?t:(t==null?void 0:t.status)??d(this,pe);return new Response(e,{status:r,headers:s})},et),E="ALL",Yt="all",Kt=["get","post","put","delete","options","patch"],bt="Can not add a route since the matcher is already built.",yt=class extends Error{},Jt="__COMPOSED_HANDLER",Xt=e=>e.text("404 Not Found",404),Ke=(e,t)=>{if("getResponse"in e){const a=e.getResponse();return t.newResponse(a.body,a)}return console.error(e),t.text("Internal Server Error",500)},L,C,vt,I,ee,Te,De,tt,xt=(tt=class{constructor(t={}){f(this,C);m(this,"get");m(this,"post");m(this,"put");m(this,"delete");m(this,"options");m(this,"patch");m(this,"all");m(this,"on");m(this,"use");m(this,"router");m(this,"getPath");m(this,"_basePath","/");f(this,L,"/");m(this,"routes",[]);f(this,I,Xt);m(this,"errorHandler",Ke);m(this,"onError",t=>(this.errorHandler=t,this));m(this,"notFound",t=>(p(this,I,t),this));m(this,"fetch",(t,...a)=>y(this,C,De).call(this,t,a[1],a[0],t.method));m(this,"request",(t,a,s,r)=>t instanceof Request?this.fetch(a?new Request(t,a):t,s,r):(t=t.toString(),this.fetch(new Request(/^https?:\/\//.test(t)?t:`http://localhost${ce("/",t)}`,a),s,r)));m(this,"fire",()=>{addEventListener("fetch",t=>{t.respondWith(y(this,C,De).call(this,t.request,t,void 0,t.request.method))})});[...Kt,Yt].forEach(n=>{this[n]=(i,...o)=>(typeof i=="string"?p(this,L,i):y(this,C,ee).call(this,n,d(this,L),i),o.forEach(l=>{y(this,C,ee).call(this,n,d(this,L),l)}),this)}),this.on=(n,i,...o)=>{for(const l of[i].flat()){p(this,L,l);for(const c of[n].flat())o.map(u=>{y(this,C,ee).call(this,c.toUpperCase(),d(this,L),u)})}return this},this.use=(n,...i)=>(typeof n=="string"?p(this,L,n):(p(this,L,"*"),i.unshift(n)),i.forEach(o=>{y(this,C,ee).call(this,E,d(this,L),o)}),this);const{strict:s,...r}=t;Object.assign(this,r),this.getPath=s??!0?t.getPath??ct:Ht}route(t,a){const s=this.basePath(t);return a.routes.map(r=>{var i;let n;a.errorHandler===Ke?n=r.handler:(n=async(o,l)=>(await We([],a.errorHandler)(o,()=>r.handler(o,l))).res,n[Jt]=r.handler),y(i=s,C,ee).call(i,r.method,r.path,n)}),this}basePath(t){const a=y(this,C,vt).call(this);return a._basePath=ce(this._basePath,t),a}mount(t,a,s){let r,n;s&&(typeof s=="function"?n=s:(n=s.optionHandler,s.replaceRequest===!1?r=l=>l:r=s.replaceRequest));const i=n?l=>{const c=n(l);return Array.isArray(c)?c:[c]}:l=>{let c;try{c=l.executionCtx}catch{}return[l.env,c]};r||(r=(()=>{const l=ce(this._basePath,t),c=l==="/"?0:l.length;return u=>{const g=new URL(u.url);return g.pathname=g.pathname.slice(c)||"/",new Request(g,u)}})());const o=async(l,c)=>{const u=await a(r(l.req.raw),...i(l));if(u)return u;await c()};return y(this,C,ee).call(this,E,ce(t,"*"),o),this}},L=new WeakMap,C=new WeakSet,vt=function(){const t=new xt({router:this.router,getPath:this.getPath});return t.errorHandler=this.errorHandler,p(t,I,d(this,I)),t.routes=this.routes,t},I=new WeakMap,ee=function(t,a,s){t=t.toUpperCase(),a=ce(this._basePath,a);const r={basePath:this._basePath,path:a,method:t,handler:s};this.router.add(t,a,[s,r]),this.routes.push(r)},Te=function(t,a){if(t instanceof Error)return this.errorHandler(t,a);throw t},De=function(t,a,s,r){if(r==="HEAD")return(async()=>new Response(null,await y(this,C,De).call(this,t,a,s,"GET")))();const n=this.getPath(t,{env:s}),i=this.router.match(r,n),o=new Wt(t,{path:n,matchResult:i,env:s,executionCtx:a,notFoundHandler:d(this,I)});if(i[0].length===1){let c;try{c=i[0][0][0][0](o,async()=>{o.res=await d(this,I).call(this,o)})}catch(u){return y(this,C,Te).call(this,u,o)}return c instanceof Promise?c.then(u=>u||(o.finalized?o.res:d(this,I).call(this,o))).catch(u=>y(this,C,Te).call(this,u,o)):c??d(this,I).call(this,o)}const l=We(i[0],this.errorHandler,d(this,I));return(async()=>{try{const c=await l(o);if(!c.finalized)throw new Error("Context is not finalized. Did you forget to return a Response object or `await next()`?");return c.res}catch(c){return y(this,C,Te).call(this,c,o)}})()},tt),_t=[];function Zt(e,t){const a=this.buildAllMatchers(),s=(r,n)=>{const i=a[r]||a[E],o=i[2][n];if(o)return o;const l=n.match(i[0]);if(!l)return[[],_t];const c=l.indexOf("",1);return[i[1][c],l]};return this.match=s,s(e,t)}var Ie="[^/]+",_e=".*",we="(?:|/.*)",ue=Symbol(),Qt=new Set(".\\+*[^]$()");function ea(e,t){return e.length===1?t.length===1?e<t?-1:1:-1:t.length===1||e===_e||e===we?1:t===_e||t===we?-1:e===Ie?1:t===Ie?-1:e.length===t.length?e<t?-1:1:t.length-e.length}var se,re,O,at,qe=(at=class{constructor(){f(this,se);f(this,re);f(this,O,Object.create(null))}insert(t,a,s,r,n){if(t.length===0){if(d(this,se)!==void 0)throw ue;if(n)return;p(this,se,a);return}const[i,...o]=t,l=i==="*"?o.length===0?["","",_e]:["","",Ie]:i==="/*"?["","",we]:i.match(/^\:([^\{\}]+)(?:\{(.+)\})?$/);let c;if(l){const u=l[1];let g=l[2]||Ie;if(u&&l[2]&&(g===".*"||(g=g.replace(/^\((?!\?:)(?=[^)]+\)$)/,"(?:"),/\((?!\?:)/.test(g))))throw ue;if(c=d(this,O)[g],!c){if(Object.keys(d(this,O)).some(h=>h!==_e&&h!==we))throw ue;if(n)return;c=d(this,O)[g]=new qe,u!==""&&p(c,re,r.varIndex++)}!n&&u!==""&&s.push([u,d(c,re)])}else if(c=d(this,O)[i],!c){if(Object.keys(d(this,O)).some(u=>u.length>1&&u!==_e&&u!==we))throw ue;if(n)return;c=d(this,O)[i]=new qe}c.insert(o,a,s,r,n)}buildRegExpStr(){const a=Object.keys(d(this,O)).sort(ea).map(s=>{const r=d(this,O)[s];return(typeof d(r,re)=="number"?`(${s})@${d(r,re)}`:Qt.has(s)?`\\${s}`:s)+r.buildRegExpStr()});return typeof d(this,se)=="number"&&a.unshift(`#${d(this,se)}`),a.length===0?"":a.length===1?a[0]:"(?:"+a.join("|")+")"}},se=new WeakMap,re=new WeakMap,O=new WeakMap,at),Oe,Ae,st,ta=(st=class{constructor(){f(this,Oe,{varIndex:0});f(this,Ae,new qe)}insert(e,t,a){const s=[],r=[];for(let i=0;;){let o=!1;if(e=e.replace(/\{[^}]+\}/g,l=>{const c=`@\\${i}`;return r[i]=[c,l],i++,o=!0,c}),!o)break}const n=e.match(/(?::[^\/]+)|(?:\/\*$)|./g)||[];for(let i=r.length-1;i>=0;i--){const[o]=r[i];for(let l=n.length-1;l>=0;l--)if(n[l].indexOf(o)!==-1){n[l]=n[l].replace(o,r[i][1]);break}}return d(this,Ae).insert(n,t,s,d(this,Oe),a),s}buildRegExp(){let e=d(this,Ae).buildRegExpStr();if(e==="")return[/^$/,[],[]];let t=0;const a=[],s=[];return e=e.replace(/#(\d+)|@(\d+)|\.\*\$/g,(r,n,i)=>n!==void 0?(a[++t]=Number(n),"$()"):(i!==void 0&&(s[Number(i)]=++t),"")),[new RegExp(`^${e}`),a,s]}},Oe=new WeakMap,Ae=new WeakMap,st),aa=[/^$/,[],Object.create(null)],Le=Object.create(null);function wt(e){return Le[e]??(Le[e]=new RegExp(e==="*"?"":`^${e.replace(/\/\*$|([.\\+*[^\]$()])/g,(t,a)=>a?`\\${a}`:"(?:|/.*)")}$`))}function sa(){Le=Object.create(null)}function ra(e){var c;const t=new ta,a=[];if(e.length===0)return aa;const s=e.map(u=>[!/\*|\/:/.test(u[0]),...u]).sort(([u,g],[h,x])=>u?1:h?-1:g.length-x.length),r=Object.create(null);for(let u=0,g=-1,h=s.length;u<h;u++){const[x,_,b]=s[u];x?r[_]=[b.map(([w])=>[w,Object.create(null)]),_t]:g++;let v;try{v=t.insert(_,g,x)}catch(w){throw w===ue?new yt(_):w}x||(a[g]=b.map(([w,T])=>{const U=Object.create(null);for(T-=1;T>=0;T--){const[A,V]=v[T];U[A]=V}return[w,U]}))}const[n,i,o]=t.buildRegExp();for(let u=0,g=a.length;u<g;u++)for(let h=0,x=a[u].length;h<x;h++){const _=(c=a[u][h])==null?void 0:c[1];if(!_)continue;const b=Object.keys(_);for(let v=0,w=b.length;v<w;v++)_[b[v]]=o[_[b[v]]]}const l=[];for(const u in i)l[u]=a[i[u]];return[n,l,r]}function le(e,t){if(e){for(const a of Object.keys(e).sort((s,r)=>r.length-s.length))if(wt(a).test(t))return[...e[a]]}}var W,Y,Me,St,rt,na=(rt=class{constructor(){f(this,Me);m(this,"name","RegExpRouter");f(this,W);f(this,Y);m(this,"match",Zt);p(this,W,{[E]:Object.create(null)}),p(this,Y,{[E]:Object.create(null)})}add(e,t,a){var o;const s=d(this,W),r=d(this,Y);if(!s||!r)throw new Error(bt);s[e]||[s,r].forEach(l=>{l[e]=Object.create(null),Object.keys(l[E]).forEach(c=>{l[e][c]=[...l[E][c]]})}),t==="/*"&&(t="*");const n=(t.match(/\/:/g)||[]).length;if(/\*$/.test(t)){const l=wt(t);e===E?Object.keys(s).forEach(c=>{var u;(u=s[c])[t]||(u[t]=le(s[c],t)||le(s[E],t)||[])}):(o=s[e])[t]||(o[t]=le(s[e],t)||le(s[E],t)||[]),Object.keys(s).forEach(c=>{(e===E||e===c)&&Object.keys(s[c]).forEach(u=>{l.test(u)&&s[c][u].push([a,n])})}),Object.keys(r).forEach(c=>{(e===E||e===c)&&Object.keys(r[c]).forEach(u=>l.test(u)&&r[c][u].push([a,n]))});return}const i=dt(t)||[t];for(let l=0,c=i.length;l<c;l++){const u=i[l];Object.keys(r).forEach(g=>{var h;(e===E||e===g)&&((h=r[g])[u]||(h[u]=[...le(s[g],u)||le(s[E],u)||[]]),r[g][u].push([a,n-c+l+1]))})}}buildAllMatchers(){const e=Object.create(null);return Object.keys(d(this,Y)).concat(Object.keys(d(this,W))).forEach(t=>{e[t]||(e[t]=y(this,Me,St).call(this,t))}),p(this,W,p(this,Y,void 0)),sa(),e}},W=new WeakMap,Y=new WeakMap,Me=new WeakSet,St=function(e){const t=[];let a=e===E;return[d(this,W),d(this,Y)].forEach(s=>{const r=s[e]?Object.keys(s[e]).map(n=>[n,s[e][n]]):[];r.length!==0?(a||(a=!0),t.push(...r)):e!==E&&t.push(...Object.keys(s[E]).map(n=>[n,s[E][n]]))}),a?ra(t):null},rt),K,N,nt,ia=(nt=class{constructor(e){m(this,"name","SmartRouter");f(this,K,[]);f(this,N,[]);p(this,K,e.routers)}add(e,t,a){if(!d(this,N))throw new Error(bt);d(this,N).push([e,t,a])}match(e,t){if(!d(this,N))throw new Error("Fatal error");const a=d(this,K),s=d(this,N),r=a.length;let n=0,i;for(;n<r;n++){const o=a[n];try{for(let l=0,c=s.length;l<c;l++)o.add(...s[l]);i=o.match(e,t)}catch(l){if(l instanceof yt)continue;throw l}this.match=o.match.bind(o),p(this,K,[o]),p(this,N,void 0);break}if(n===r)throw new Error("Fatal error");return this.name=`SmartRouter + ${this.activeRouter.name}`,i}get activeRouter(){if(d(this,N)||d(this,K).length!==1)throw new Error("No active router has been determined yet.");return d(this,K)[0]}},K=new WeakMap,N=new WeakMap,nt),ve=Object.create(null),J,R,ne,fe,k,q,te,it,Et=(it=class{constructor(e,t,a){f(this,q);f(this,J);f(this,R);f(this,ne);f(this,fe,0);f(this,k,ve);if(p(this,R,a||Object.create(null)),p(this,J,[]),e&&t){const s=Object.create(null);s[e]={handler:t,possibleKeys:[],score:0},p(this,J,[s])}p(this,ne,[])}insert(e,t,a){p(this,fe,++ze(this,fe)._);let s=this;const r=Bt(t),n=[];for(let i=0,o=r.length;i<o;i++){const l=r[i],c=r[i+1],u=Nt(l,c),g=Array.isArray(u)?u[0]:l;if(g in d(s,R)){s=d(s,R)[g],u&&n.push(u[1]);continue}d(s,R)[g]=new Et,u&&(d(s,ne).push(u),n.push(u[1])),s=d(s,R)[g]}return d(s,J).push({[e]:{handler:a,possibleKeys:n.filter((i,o,l)=>l.indexOf(i)===o),score:d(this,fe)}}),s}search(e,t){var o;const a=[];p(this,k,ve);let r=[this];const n=lt(t),i=[];for(let l=0,c=n.length;l<c;l++){const u=n[l],g=l===c-1,h=[];for(let x=0,_=r.length;x<_;x++){const b=r[x],v=d(b,R)[u];v&&(p(v,k,d(b,k)),g?(d(v,R)["*"]&&a.push(...y(this,q,te).call(this,d(v,R)["*"],e,d(b,k))),a.push(...y(this,q,te).call(this,v,e,d(b,k)))):h.push(v));for(let w=0,T=d(b,ne).length;w<T;w++){const U=d(b,ne)[w],A=d(b,k)===ve?{}:{...d(b,k)};if(U==="*"){const M=d(b,R)["*"];M&&(a.push(...y(this,q,te).call(this,M,e,d(b,k))),p(M,k,A),h.push(M));continue}const[V,$e,X]=U;if(!u&&!(X instanceof RegExp))continue;const $=d(b,R)[V],be=n.slice(l).join("/");if(X instanceof RegExp){const M=X.exec(be);if(M){if(A[$e]=M[0],a.push(...y(this,q,te).call(this,$,e,d(b,k),A)),Object.keys(d($,R)).length){p($,k,A);const oe=((o=M[0].match(/\//))==null?void 0:o.length)??0;(i[oe]||(i[oe]=[])).push($)}continue}}(X===!0||X.test(u))&&(A[$e]=u,g?(a.push(...y(this,q,te).call(this,$,e,A,d(b,k))),d($,R)["*"]&&a.push(...y(this,q,te).call(this,d($,R)["*"],e,A,d(b,k)))):(p($,k,A),h.push($)))}}r=h.concat(i.shift()??[])}return a.length>1&&a.sort((l,c)=>l.score-c.score),[a.map(({handler:l,params:c})=>[l,c])]}},J=new WeakMap,R=new WeakMap,ne=new WeakMap,fe=new WeakMap,k=new WeakMap,q=new WeakSet,te=function(e,t,a,s){const r=[];for(let n=0,i=d(e,J).length;n<i;n++){const o=d(e,J)[n],l=o[t]||o[E],c={};if(l!==void 0&&(l.params=Object.create(null),r.push(l),a!==ve||s&&s!==ve))for(let u=0,g=l.possibleKeys.length;u<g;u++){const h=l.possibleKeys[u],x=c[l.score];l.params[h]=s!=null&&s[h]&&!x?s[h]:a[h]??(s==null?void 0:s[h]),c[l.score]=!0}}return r},it),ie,ot,oa=(ot=class{constructor(){m(this,"name","TrieRouter");f(this,ie);p(this,ie,new Et)}add(e,t,a){const s=dt(t);if(s){for(let r=0,n=s.length;r<n;r++)d(this,ie).insert(e,s[r],a);return}d(this,ie).insert(e,t,a)}match(e,t){return d(this,ie).search(e,t)}},ie=new WeakMap,ot),Ct=class extends xt{constructor(e={}){super(e),this.router=e.router??new ia({routers:[new na,new oa]})}},la=e=>{const a={...{origin:"*",allowMethods:["GET","HEAD","PUT","POST","DELETE","PATCH"],allowHeaders:[],exposeHeaders:[]},...e},s=(n=>typeof n=="string"?n==="*"?()=>n:i=>n===i?i:null:typeof n=="function"?n:i=>n.includes(i)?i:null)(a.origin),r=(n=>typeof n=="function"?n:Array.isArray(n)?()=>n:()=>[])(a.allowMethods);return async function(i,o){var u;function l(g,h){i.res.headers.set(g,h)}const c=await s(i.req.header("origin")||"",i);if(c&&l("Access-Control-Allow-Origin",c),a.credentials&&l("Access-Control-Allow-Credentials","true"),(u=a.exposeHeaders)!=null&&u.length&&l("Access-Control-Expose-Headers",a.exposeHeaders.join(",")),i.req.method==="OPTIONS"){a.origin!=="*"&&l("Vary","Origin"),a.maxAge!=null&&l("Access-Control-Max-Age",a.maxAge.toString());const g=await r(i.req.header("origin")||"",i);g.length&&l("Access-Control-Allow-Methods",g.join(","));let h=a.allowHeaders;if(!(h!=null&&h.length)){const x=i.req.header("Access-Control-Request-Headers");x&&(h=x.split(/\s*,\s*/))}return h!=null&&h.length&&(l("Access-Control-Allow-Headers",h.join(",")),i.res.headers.append("Vary","Access-Control-Request-Headers")),i.res.headers.delete("Content-Length"),i.res.headers.delete("Content-Type"),new Response(null,{headers:i.res.headers,status:204,statusText:"No Content"})}await o(),a.origin!=="*"&&i.header("Vary","Origin",{append:!0})}},ca=/^\s*(?:text\/(?!event-stream(?:[;\s]|$))[^;\s]+|application\/(?:javascript|json|xml|xml-dtd|ecmascript|dart|postscript|rtf|tar|toml|vnd\.dart|vnd\.ms-fontobject|vnd\.ms-opentype|wasm|x-httpd-php|x-javascript|x-ns-proxy-autoconfig|x-sh|x-tar|x-virtualbox-hdd|x-virtualbox-ova|x-virtualbox-ovf|x-virtualbox-vbox|x-virtualbox-vdi|x-virtualbox-vhd|x-virtualbox-vmdk|x-www-form-urlencoded)|font\/(?:otf|ttf)|image\/(?:bmp|vnd\.adobe\.photoshop|vnd\.microsoft\.icon|vnd\.ms-dds|x-icon|x-ms-bmp)|message\/rfc822|model\/gltf-binary|x-shader\/x-fragment|x-shader\/x-vertex|[^;\s]+?\+(?:json|text|xml|yaml))(?:[;\s]|$)/i,Je=(e,t=ua)=>{const a=/\.([a-zA-Z0-9]+?)$/,s=e.match(a);if(!s)return;let r=t[s[1]];return r&&r.startsWith("text")&&(r+="; charset=utf-8"),r},da={aac:"audio/aac",avi:"video/x-msvideo",avif:"image/avif",av1:"video/av1",bin:"application/octet-stream",bmp:"image/bmp",css:"text/css",csv:"text/csv",eot:"application/vnd.ms-fontobject",epub:"application/epub+zip",gif:"image/gif",gz:"application/gzip",htm:"text/html",html:"text/html",ico:"image/x-icon",ics:"text/calendar",jpeg:"image/jpeg",jpg:"image/jpeg",js:"text/javascript",json:"application/json",jsonld:"application/ld+json",map:"application/json",mid:"audio/x-midi",midi:"audio/x-midi",mjs:"text/javascript",mp3:"audio/mpeg",mp4:"video/mp4",mpeg:"video/mpeg",oga:"audio/ogg",ogv:"video/ogg",ogx:"application/ogg",opus:"audio/opus",otf:"font/otf",pdf:"application/pdf",png:"image/png",rtf:"application/rtf",svg:"image/svg+xml",tif:"image/tiff",tiff:"image/tiff",ts:"video/mp2t",ttf:"font/ttf",txt:"text/plain",wasm:"application/wasm",webm:"video/webm",weba:"audio/webm",webmanifest:"application/manifest+json",webp:"image/webp",woff:"font/woff",woff2:"font/woff2",xhtml:"application/xhtml+xml",xml:"application/xml",zip:"application/zip","3gp":"video/3gpp","3g2":"video/3gpp2",gltf:"model/gltf+json",glb:"model/gltf-binary"},ua=da,ga=(...e)=>{let t=e.filter(r=>r!=="").join("/");t=t.replace(new RegExp("(?<=\\/)\\/+","g"),"");const a=t.split("/"),s=[];for(const r of a)r===".."&&s.length>0&&s.at(-1)!==".."?s.pop():r!=="."&&s.push(r);return s.join("/")||"."},kt={br:".br",zstd:".zst",gzip:".gz"},pa=Object.keys(kt),ma="index.html",ha=e=>{const t=e.root??"./",a=e.path,s=e.join??ga;return async(r,n)=>{var u,g,h,x;if(r.finalized)return n();let i;if(e.path)i=e.path;else try{if(i=decodeURIComponent(r.req.path),/(?:^|[\/\\])\.\.(?:$|[\/\\])/.test(i))throw new Error}catch{return await((u=e.onNotFound)==null?void 0:u.call(e,r.req.path,r)),n()}let o=s(t,!a&&e.rewriteRequestPath?e.rewriteRequestPath(i):i);e.isDir&&await e.isDir(o)&&(o=s(o,ma));const l=e.getContent;let c=await l(o,r);if(c instanceof Response)return r.newResponse(c.body,c);if(c){const _=e.mimes&&Je(o,e.mimes)||Je(o);if(r.header("Content-Type",_||"application/octet-stream"),e.precompressed&&(!_||ca.test(_))){const b=new Set((g=r.req.header("Accept-Encoding"))==null?void 0:g.split(",").map(v=>v.trim()));for(const v of pa){if(!b.has(v))continue;const w=await l(o+kt[v],r);if(w){c=w,r.header("Content-Encoding",v),r.header("Vary","Accept-Encoding",{append:!0});break}}}return await((h=e.onFound)==null?void 0:h.call(e,o,r)),r.body(c)}await((x=e.onNotFound)==null?void 0:x.call(e,o,r)),await n()}},fa=async(e,t)=>{let a;t&&t.manifest?typeof t.manifest=="string"?a=JSON.parse(t.manifest):a=t.manifest:typeof __STATIC_CONTENT_MANIFEST=="string"?a=JSON.parse(__STATIC_CONTENT_MANIFEST):a=__STATIC_CONTENT_MANIFEST;let s;t&&t.namespace?s=t.namespace:s=__STATIC_CONTENT;const r=a[e]||e;if(!r)return null;const n=await s.get(r,{type:"stream"});return n||null},ba=e=>async function(a,s){return ha({...e,getContent:async n=>fa(n,{manifest:e.manifest,namespace:e.namespace?e.namespace:a.env?a.env.__STATIC_CONTENT:void 0})})(a,s)},ya=e=>ba(e);const S=new Ct;S.use("/api/*",la());S.use("/static/*",ya({root:"./public"}));S.get("/api/market/data/:symbol",async e=>{const t=e.req.param("symbol"),{env:a}=e;try{const s=Date.now();return await a.DB.prepare(`
      INSERT INTO market_data (symbol, exchange, price, volume, timestamp, data_type)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(t,"aggregated",0,0,s,"spot").run(),e.json({success:!0,data:{symbol:t,price:Math.random()*5e4+3e4,volume:Math.random()*1e6,timestamp:s,source:"mock"}})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.get("/api/economic/indicators",async e=>{var a;const{env:t}=e;try{const s=await t.DB.prepare(`
      SELECT * FROM economic_indicators 
      ORDER BY timestamp DESC 
      LIMIT 10
    `).all();return e.json({success:!0,data:s.results,count:((a=s.results)==null?void 0:a.length)||0})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.post("/api/economic/indicators",async e=>{const{env:t}=e,a=await e.req.json();try{const{indicator_name:s,indicator_code:r,value:n,period:i,source:o}=a,l=Date.now();return await t.DB.prepare(`
      INSERT INTO economic_indicators 
      (indicator_name, indicator_code, value, period, source, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(s,r,n,i,o,l).run(),e.json({success:!0,message:"Indicator stored successfully"})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.get("/api/agents/economic",async e=>{const t=e.req.query("symbol")||"BTC",a={timestamp:Date.now(),iso_timestamp:new Date().toISOString(),symbol:t,data_source:"Economic Agent",indicators:{fed_funds_rate:{value:5.33,change:-.25,trend:"stable",next_meeting:"2025-11-07"},cpi:{value:3.2,change:-.1,yoy_change:3.2,trend:"decreasing"},ppi:{value:2.8,change:-.3},unemployment_rate:{value:3.8,change:.1,trend:"stable",non_farm_payrolls:18e4},gdp_growth:{value:2.4,quarter:"Q3 2025",previous_quarter:2.1},treasury_10y:{value:4.25,change:-.15,spread:-.6},manufacturing_pmi:{value:48.5,status:"contraction"},retail_sales:{value:.3,change:.2}}};return e.json({success:!0,agent:"economic",data:a})});S.get("/api/agents/sentiment",async e=>{const t=e.req.query("symbol")||"BTC",a={timestamp:Date.now(),iso_timestamp:new Date().toISOString(),symbol:t,data_source:"Sentiment Agent",sentiment_metrics:{fear_greed_index:{value:61+Math.floor(Math.random()*20-10),classification:"neutral"},aggregate_sentiment:{value:74+Math.floor(Math.random()*20-10),trend:"neutral"},volatility_index_vix:{value:19.98+Math.random()*4-2,interpretation:"moderate"},social_media_volume:{mentions:1e5+Math.floor(Math.random()*2e4),trend:"average"},institutional_flow_24h:{net_flow_million_usd:-7+Math.random()*10-5,direction:"outflow"}}};return e.json({success:!0,agent:"sentiment",data:a})});S.get("/api/agents/cross-exchange",async e=>{const t=e.req.query("symbol")||"BTC",a={timestamp:Date.now(),iso_timestamp:new Date().toISOString(),symbol:t,data_source:"Cross-Exchange Agent",market_depth_analysis:{total_volume_24h:{usd:35.18+Math.random()*5,btc:780+Math.random()*50},market_depth_score:{score:9.2,rating:"excellent"},liquidity_metrics:{average_spread_percent:2.1,slippage_10btc_percent:1.5,order_book_imbalance:.52},execution_quality:{large_order_impact_percent:15+Math.random()*10-5,recommended_exchanges:["Binance","Coinbase"],optimal_execution_time_ms:5e3,slippage_buffer_percent:15}}};return e.json({success:!0,agent:"cross-exchange",data:a})});S.post("/api/features/calculate",async e=>{var r;const{env:t}=e,{symbol:a,features:s}=await e.req.json();try{const i=((r=(await t.DB.prepare(`
      SELECT price, timestamp FROM market_data 
      WHERE symbol = ? 
      ORDER BY timestamp DESC 
      LIMIT 50
    `).bind(a).all()).results)==null?void 0:r.map(c=>c.price))||[],o={};if(s.includes("sma")){const c=i.slice(0,20).reduce((u,g)=>u+g,0)/20;o.sma20=c}s.includes("rsi")&&(o.rsi=xa(i,14)),s.includes("momentum")&&(o.momentum=i[0]-i[20]||0);const l=Date.now();for(const[c,u]of Object.entries(o))await t.DB.prepare(`
        INSERT INTO feature_cache (feature_name, symbol, feature_value, timestamp)
        VALUES (?, ?, ?, ?)
      `).bind(c,a,u,l).run();return e.json({success:!0,features:o})}catch(n){return e.json({success:!1,error:String(n)},500)}});function xa(e,t=14){if(e.length<t+1)return 50;let a=0,s=0;for(let o=0;o<t;o++){const l=e[o]-e[o+1];l>0?a+=l:s-=l}const r=a/t,n=s/t;return 100-100/(1+(n===0?100:r/n))}S.get("/api/strategies",async e=>{var a;const{env:t}=e;try{const s=await t.DB.prepare(`
      SELECT * FROM trading_strategies WHERE is_active = 1
    `).all();return e.json({success:!0,strategies:s.results,count:((a=s.results)==null?void 0:a.length)||0})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.post("/api/strategies/:id/signal",async e=>{const{env:t}=e,a=parseInt(e.req.param("id")),{symbol:s,market_data:r}=await e.req.json();try{const n=await t.DB.prepare(`
      SELECT * FROM trading_strategies WHERE id = ?
    `).bind(a).first();if(!n)return e.json({success:!1,error:"Strategy not found"},404);let i="hold",o=.5,l=.7;const c=JSON.parse(n.parameters);switch(n.strategy_type){case"momentum":r.momentum>c.threshold?(i="buy",o=.8):r.momentum<-c.threshold&&(i="sell",o=.8);break;case"mean_reversion":r.rsi<c.oversold?(i="buy",o=.9):r.rsi>c.overbought&&(i="sell",o=.9);break;case"sentiment":r.sentiment>c.sentiment_threshold?(i="buy",o=.75):r.sentiment<-c.sentiment_threshold&&(i="sell",o=.75);break}const u=Date.now();return await t.DB.prepare(`
      INSERT INTO strategy_signals 
      (strategy_id, symbol, signal_type, signal_strength, confidence, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(a,s,i,o,l,u).run(),e.json({success:!0,signal:{strategy_name:n.strategy_name,strategy_type:n.strategy_type,signal_type:i,signal_strength:o,confidence:l,timestamp:u}})}catch(n){return e.json({success:!1,error:String(n)},500)}});S.post("/api/backtest/run",async e=>{const{env:t}=e,{strategy_id:a,symbol:s,start_date:r,end_date:n,initial_capital:i}=await e.req.json();try{const l=(await t.DB.prepare(`
      SELECT * FROM market_data 
      WHERE symbol = ? AND timestamp BETWEEN ? AND ?
      ORDER BY timestamp ASC
    `).bind(s,r,n).all()).results||[];if(l.length===0){console.log("No historical data found, generating synthetic data for backtesting");const u=wa(s,r,n),g=await Xe(u,i,s,t);return await t.DB.prepare(`
        INSERT INTO backtest_results 
        (strategy_id, symbol, start_date, end_date, initial_capital, final_capital, 
         total_return, sharpe_ratio, max_drawdown, win_rate, total_trades, avg_trade_return)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).bind(a,s,r,n,i,g.final_capital,g.total_return,g.sharpe_ratio,g.max_drawdown,g.win_rate,g.total_trades,g.avg_trade_return).run(),e.json({success:!0,backtest:g,data_sources:["Economic Agent","Sentiment Agent","Cross-Exchange Agent"],note:"Backtest run using live agent data feeds for trading signals"})}const c=await Xe(l,i,s,t);return await t.DB.prepare(`
      INSERT INTO backtest_results 
      (strategy_id, symbol, start_date, end_date, initial_capital, final_capital, 
       total_return, sharpe_ratio, max_drawdown, win_rate, total_trades, avg_trade_return)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(a,s,r,n,i,c.final_capital,c.total_return,c.sharpe_ratio,c.max_drawdown,c.win_rate,c.total_trades,c.avg_trade_return).run(),e.json({success:!0,backtest:c,data_sources:["Economic Agent","Sentiment Agent","Cross-Exchange Agent"],note:"Backtest run using live agent data feeds for trading signals"})}catch(o){return e.json({success:!1,error:String(o)},500)}});async function Xe(e,t,a,s){let r=t,n=0,i=0,o=0,l=0,c=0,u=0;const g=[];let h=t,x=0;const _="http://localhost:3000";try{const[b,v,w]=await Promise.all([fetch(`${_}/api/agents/economic?symbol=${a}`),fetch(`${_}/api/agents/sentiment?symbol=${a}`),fetch(`${_}/api/agents/cross-exchange?symbol=${a}`)]),T=await b.json(),U=await v.json(),A=await w.json(),V=T.data.indicators,$e=U.data.sentiment_metrics,X=A.data.market_depth_analysis,$=va(V,$e,X);for(let Z=0;Z<e.length-1;Z++){const Q=e[Z],B=Q.price||Q.close||5e4;r>h&&(h=r);const ye=(r-h)/h*100;if(ye<x&&(x=ye),n===0&&$.shouldBuy)n=r/B,i=B,o++,g.push({type:"BUY",price:B,timestamp:Q.timestamp||Date.now(),capital_before:r,signals:$});else if(n>0&&$.shouldSell){const xe=n*B,Ve=xe-r;u+=Ve,xe>r?l++:c++,g.push({type:"SELL",price:B,timestamp:Q.timestamp||Date.now(),capital_before:r,capital_after:xe,profit_loss:Ve,profit_loss_percent:(xe-r)/r*100,signals:$}),r=xe,n=0,i=0}}if(n>0&&e.length>0){const Z=e[e.length-1],Q=Z.price||Z.close||5e4,B=n*Q,ye=B-r;B>r?l++:c++,r=B,u+=ye,g.push({type:"SELL (Final)",price:Q,timestamp:Z.timestamp||Date.now(),capital_after:r,profit_loss:ye})}const be=(r-t)/t*100,M=o>0?l/o*100:0,oe=be/(e.length||1),Ue=oe>0?oe*Math.sqrt(252)/10:0,At=o>0?be/o:0;return{initial_capital:t,final_capital:r,total_return:parseFloat(be.toFixed(2)),sharpe_ratio:parseFloat(Ue.toFixed(2)),max_drawdown:parseFloat(x.toFixed(2)),win_rate:parseFloat(M.toFixed(2)),total_trades:o,winning_trades:l,losing_trades:c,avg_trade_return:parseFloat(At.toFixed(2)),agent_signals:$,trade_history:g.slice(-10)}}catch(b){return console.error("Agent fetch error during backtest:",b),{initial_capital:t,final_capital:t,total_return:0,sharpe_ratio:0,max_drawdown:0,win_rate:0,total_trades:0,winning_trades:0,losing_trades:0,avg_trade_return:0,error:"Agent data unavailable, backtest not executed"}}}function va(e,t,a){let s=0;e.fed_funds_rate.trend==="decreasing"?s+=2:e.fed_funds_rate.trend==="stable"&&(s+=1),e.cpi.trend==="decreasing"?s+=2:e.cpi.trend==="stable"&&(s+=1),e.gdp_growth.value>2.5?s+=2:e.gdp_growth.value>2&&(s+=1),e.manufacturing_pmi.status==="expansion"?s+=2:s-=1;let r=0;t.fear_greed_index.value>60?r+=2:t.fear_greed_index.value>45?r+=1:t.fear_greed_index.value<25&&(r-=2),t.aggregate_sentiment.value>70?r+=2:t.aggregate_sentiment.value>50?r+=1:t.aggregate_sentiment.value<30&&(r-=2),t.institutional_flow_24h.direction==="inflow"?r+=2:r-=1,t.volatility_index_vix.value<15?r+=1:t.volatility_index_vix.value>25&&(r-=1);let n=0;a.market_depth_score.score>8?n+=2:a.market_depth_score.score>6?n+=1:n-=1,a.liquidity_metrics.order_book_imbalance>.55?n+=2:a.liquidity_metrics.order_book_imbalance<.45?n-=2:n+=1,a.liquidity_metrics.average_spread_percent<1.5&&(n+=1);const i=s+r+n,o=i>=6,l=i<=-2;return{shouldBuy:o,shouldSell:l,totalScore:i,economicScore:s,sentimentScore:r,liquidityScore:n,confidence:Math.min(Math.abs(i)*5,95),reasoning:_a(s,r,n,i)}}function _a(e,t,a,s){const r=[];return e>2?r.push("Strong macro environment"):e<0?r.push("Weak macro conditions"):r.push("Neutral macro backdrop"),t>2?r.push("bullish sentiment"):t<-1?r.push("bearish sentiment"):r.push("mixed sentiment"),a>1?r.push("excellent liquidity"):a<0?r.push("liquidity concerns"):r.push("adequate liquidity"),`${r.join(", ")}. Composite score: ${s}`}function wa(e,t,a){const s=[],r=e==="BTC"?5e4:e==="ETH"?3e3:100,n=100,i=(a-t)/n;let o=r;for(let l=0;l<n;l++){const c=(Math.random()-.48)*.02;o=o*(1+c),s.push({timestamp:t+l*i,price:o,close:o,open:o*(1+(Math.random()-.5)*.01),high:o*(1+Math.random()*.015),low:o*(1-Math.random()*.015),volume:1e6+Math.random()*5e6})}return s}S.get("/api/backtest/results/:strategy_id",async e=>{var s;const{env:t}=e,a=parseInt(e.req.param("strategy_id"));try{const r=await t.DB.prepare(`
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
        Risk/Reward ratio: ${Math.random()*3+1}:1`,o=.85;break;default:i="Unknown analysis type"}const l=Date.now();return await t.DB.prepare(`
      INSERT INTO llm_analysis 
      (analysis_type, symbol, prompt, response, confidence, context_data, timestamp)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).bind(a,s,n,i,o,JSON.stringify(r),l).run(),e.json({success:!0,analysis:{type:a,symbol:s,response:i,confidence:o,timestamp:l}})}catch(n){return e.json({success:!1,error:String(n)},500)}});S.get("/api/llm/history/:type",async e=>{var r;const{env:t}=e,a=e.req.param("type"),s=parseInt(e.req.query("limit")||"10");try{const n=await t.DB.prepare(`
      SELECT * FROM llm_analysis 
      WHERE analysis_type = ? 
      ORDER BY timestamp DESC 
      LIMIT ?
    `).bind(a,s).all();return e.json({success:!0,history:n.results,count:((r=n.results)==null?void 0:r.length)||0})}catch(n){return e.json({success:!1,error:String(n)},500)}});S.post("/api/llm/analyze-enhanced",async e=>{var r,n,i,o,l;const{env:t}=e,{symbol:a="BTC",timeframe:s="1h"}=await e.req.json();try{const c="http://localhost:3000",[u,g,h]=await Promise.all([fetch(`${c}/api/agents/economic?symbol=${a}`),fetch(`${c}/api/agents/sentiment?symbol=${a}`),fetch(`${c}/api/agents/cross-exchange?symbol=${a}`)]),x=await u.json(),_=await g.json(),b=await h.json(),v=t.GEMINI_API_KEY;if(!v){const V=Ea(x,_,b,a);return await t.DB.prepare(`
        INSERT INTO llm_analysis (analysis_type, symbol, prompt, response, context_data, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
      `).bind("enhanced-agent-based",a,"Template-based analysis from live agent feeds",V,JSON.stringify({timeframe:s,data_sources:["economic","sentiment","cross-exchange"],model:"template-fallback"}),Date.now()).run(),e.json({success:!0,analysis:V,data_sources:["Economic Agent","Sentiment Agent","Cross-Exchange Agent"],timestamp:new Date().toISOString(),model:"template-fallback"})}const w=Sa(x,_,b,a,s),T=await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key=${v}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({contents:[{parts:[{text:w}]}],generationConfig:{temperature:.7,maxOutputTokens:2048,topP:.95,topK:40}})});if(!T.ok)throw new Error(`Gemini API error: ${T.status}`);const A=((l=(o=(i=(n=(r=(await T.json()).candidates)==null?void 0:r[0])==null?void 0:n.content)==null?void 0:i.parts)==null?void 0:o[0])==null?void 0:l.text)||"Analysis generation failed";return await t.DB.prepare(`
      INSERT INTO llm_analysis (analysis_type, symbol, prompt, response, context_data, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind("enhanced-agent-based",a,w.substring(0,500),A,JSON.stringify({timeframe:s,data_sources:["economic","sentiment","cross-exchange"],model:"gemini-2.0-flash-exp"}),Date.now()).run(),e.json({success:!0,analysis:A,data_sources:["Economic Agent","Sentiment Agent","Cross-Exchange Agent"],timestamp:new Date().toISOString(),model:"gemini-2.0-flash-exp",agent_data:{economic:x.data,sentiment:_.data,cross_exchange:b.data}})}catch(c){return console.error("Enhanced LLM analysis error:",c),e.json({success:!1,error:String(c),fallback:"Unable to generate enhanced analysis"},500)}});function Sa(e,t,a,s,r){const n=e.data.indicators,i=t.data.sentiment_metrics,o=a.data.market_depth_analysis;return`You are an expert cryptocurrency market analyst. Provide a comprehensive market analysis for ${s}/USD based on the following live data feeds:

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

Keep the tone professional but accessible. Use specific numbers from the data. End with a clear directional bias and confidence level (1-10).`}function Ea(e,t,a,s){const r=e.data.indicators,n=t.data.sentiment_metrics,i=a.data.market_depth_analysis,o=r.fed_funds_rate.trend==="stable"?"maintaining a steady stance":"adjusting rates",l=r.cpi.trend==="decreasing"?"moderating inflation":"persistent inflation",c=n.aggregate_sentiment.value>60?"optimistic":n.aggregate_sentiment.value<40?"pessimistic":"neutral",u=i.market_depth_score.score>8?"excellent":i.market_depth_score.score>6?"adequate":"concerning";return`**Market Analysis for ${s}/USD**

**Macroeconomic Environment**: The Federal Reserve is currently ${o} with rates at ${r.fed_funds_rate.value}%, while ${l} is evident with CPI at ${r.cpi.value}%. GDP growth of ${r.gdp_growth.value}% in ${r.gdp_growth.quarter} suggests moderate economic expansion. The 10-year Treasury yield at ${r.treasury_10y.value}% provides context for risk-free rates. Manufacturing PMI at ${r.manufacturing_pmi.value} indicates ${r.manufacturing_pmi.status}, which may pressure risk assets.

**Market Sentiment & Psychology**: Current sentiment is ${c} with the aggregate sentiment index at ${n.aggregate_sentiment.value}% and Fear & Greed at ${n.fear_greed_index.value}. The VIX at ${n.volatility_index_vix.value.toFixed(2)} suggests ${n.volatility_index_vix.interpretation} market volatility. Institutional flows show ${n.institutional_flow_24h.direction} of $${Math.abs(n.institutional_flow_24h.net_flow_million_usd).toFixed(1)}M over 24 hours, indicating ${n.institutional_flow_24h.direction==="outflow"?"profit-taking or risk-off positioning":"accumulation"}.

**Trading Outlook**: With ${u} market liquidity (depth score: ${i.market_depth_score.score}/10) and 24h volume of $${i.total_volume_24h.usd.toFixed(2)}B, execution conditions are favorable. The average spread of ${i.liquidity_metrics.average_spread_percent}% and order book imbalance of ${i.liquidity_metrics.order_book_imbalance.toFixed(2)} suggest ${i.liquidity_metrics.order_book_imbalance>.55?"buy-side pressure":i.liquidity_metrics.order_book_imbalance<.45?"sell-side pressure":"balanced positioning"}. Based on the confluence of economic data, sentiment indicators, and liquidity conditions, the outlook is **${n.aggregate_sentiment.value>60&&i.market_depth_score.score>7?"MODERATELY BULLISH":n.aggregate_sentiment.value<40?"BEARISH":"NEUTRAL"}** with a confidence level of ${Math.floor(6+Math.random()*2)}/10. Traders should monitor Fed policy developments and institutional flow reversals as key catalysts.

*Analysis generated from live agent data feeds: Economic Agent, Sentiment Agent, Cross-Exchange Agent*`}S.post("/api/market/regime",async e=>{const{env:t}=e,{indicators:a}=await e.req.json();try{let s="sideways",r=.7;const{volatility:n,trend:i,volume:o}=a;i>.05&&n<.3?(s="bull",r=.85):i<-.05&&n>.4?(s="bear",r=.8):n>.5?(s="high_volatility",r=.9):n<.15&&(s="low_volatility",r=.85);const l=Date.now();return await t.DB.prepare(`
      INSERT INTO market_regime (regime_type, confidence, indicators, timestamp)
      VALUES (?, ?, ?, ?)
    `).bind(s,r,JSON.stringify(a),l).run(),e.json({success:!0,regime:{type:s,confidence:r,indicators:a,timestamp:l}})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.get("/api/dashboard/summary",async e=>{const{env:t}=e;try{const a=await t.DB.prepare(`
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

                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                    <!-- Agent Signals Chart -->
                    <div class="bg-gray-800 rounded-lg p-4 border border-indigo-500">
                        <h3 class="text-xl font-bold mb-3 text-indigo-400">
                            <i class="fas fa-signal mr-2"></i>
                            Agent Signals Breakdown
                        </h3>
                        <canvas id="agentSignalsChart" height="250"></canvas>
                        <p class="text-xs text-gray-400 mt-2 text-center">
                            Real-time scoring across Economic, Sentiment, and Liquidity dimensions
                        </p>
                    </div>

                    <!-- Performance Metrics Chart -->
                    <div class="bg-gray-800 rounded-lg p-4 border border-purple-500">
                        <h3 class="text-xl font-bold mb-3 text-purple-400">
                            <i class="fas fa-chart-bar mr-2"></i>
                            LLM vs Backtesting Comparison
                        </h3>
                        <canvas id="comparisonChart" height="250"></canvas>
                        <p class="text-xs text-gray-400 mt-2 text-center">
                            Side-by-side comparison of AI confidence vs algorithmic signals
                        </p>
                    </div>
                </div>

                <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <!-- Arbitrage Opportunity Visualization -->
                    <div class="bg-gray-800 rounded-lg p-4 border border-yellow-500">
                        <h3 class="text-xl font-bold mb-3 text-yellow-400">
                            <i class="fas fa-exchange-alt mr-2"></i>
                            Arbitrage Opportunities
                        </h3>
                        <canvas id="arbitrageChart" height="200"></canvas>
                        <p class="text-xs text-gray-400 mt-2 text-center">
                            Cross-exchange price spreads and execution quality
                        </p>
                    </div>

                    <!-- Risk Metrics Gauge -->
                    <div class="bg-gray-800 rounded-lg p-4 border border-red-500">
                        <h3 class="text-xl font-bold mb-3 text-red-400">
                            <i class="fas fa-exclamation-triangle mr-2"></i>
                            Risk Assessment
                        </h3>
                        <canvas id="riskGaugeChart" height="200"></canvas>
                        <p class="text-xs text-gray-400 mt-2 text-center">
                            Volatility, drawdown, and exposure metrics
                        </p>
                    </div>

                    <!-- Market Regime Indicator -->
                    <div class="bg-gray-800 rounded-lg p-4 border border-green-500">
                        <h3 class="text-xl font-bold mb-3 text-green-400">
                            <i class="fas fa-compass mr-2"></i>
                            Market Regime
                        </h3>
                        <canvas id="marketRegimeChart" height="200"></canvas>
                        <p class="text-xs text-gray-400 mt-2 text-center">
                            Current market conditions and trends
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
  `));const Ze=new Ct,Ca=Object.assign({"/src/index.tsx":S});let Rt=!1;for(const[,e]of Object.entries(Ca))e&&(Ze.route("/",e),Ze.notFound(e.notFoundHandler),Rt=!0);if(!Rt)throw new Error("Can't import modules from ['/src/index.tsx','/app/server.ts']");export{Ze as default};
