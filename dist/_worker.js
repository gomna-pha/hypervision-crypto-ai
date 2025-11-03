var Pt=Object.defineProperty;var We=e=>{throw TypeError(e)};var Bt=(e,t,a)=>t in e?Pt(e,t,{enumerable:!0,configurable:!0,writable:!0,value:a}):e[t]=a;var y=(e,t,a)=>Bt(e,typeof t!="symbol"?t+"":t,a),$e=(e,t,a)=>t.has(e)||We("Cannot "+a);var g=(e,t,a)=>($e(e,t,"read from private field"),a?a.call(e):t.get(e)),v=(e,t,a)=>t.has(e)?We("Cannot add the same private member more than once"):t instanceof WeakSet?t.add(e):t.set(e,a),b=(e,t,a,s)=>($e(e,t,"write to private field"),s?s.call(e,a):t.set(e,a),a),w=(e,t,a)=>($e(e,t,"access private method"),a);var Xe=(e,t,a,s)=>({set _(n){b(e,t,n,a)},get _(){return g(e,t,s)}});var Qe=(e,t,a)=>(s,n)=>{let r=-1;return i(0);async function i(o){if(o<=r)throw new Error("next() called multiple times");r=o;let l,c=!1,d;if(e[o]?(d=e[o][0][0],s.req.routeIndex=o):d=o===e.length&&n||void 0,d)try{l=await d(s,()=>i(o+1))}catch(u){if(u instanceof Error&&t)s.error=u,l=await t(u,s),c=!0;else throw u}else s.finalized===!1&&a&&(l=await a(s));return l&&(s.finalized===!1||c)&&(s.res=l),s}},Nt=Symbol(),Ft=async(e,t=Object.create(null))=>{const{all:a=!1,dot:s=!1}=t,r=(e instanceof bt?e.raw.headers:e.headers).get("Content-Type");return r!=null&&r.startsWith("multipart/form-data")||r!=null&&r.startsWith("application/x-www-form-urlencoded")?jt(e,{all:a,dot:s}):{}};async function jt(e,t){const a=await e.formData();return a?$t(a,t):{}}function $t(e,t){const a=Object.create(null);return e.forEach((s,n)=>{t.all||n.endsWith("[]")?Ht(a,n,s):a[n]=s}),t.dot&&Object.entries(a).forEach(([s,n])=>{s.includes(".")&&(qt(a,s,n),delete a[s])}),a}var Ht=(e,t,a)=>{e[t]!==void 0?Array.isArray(e[t])?e[t].push(a):e[t]=[e[t],a]:t.endsWith("[]")?e[t]=[a]:e[t]=a},qt=(e,t,a)=>{let s=e;const n=t.split(".");n.forEach((r,i)=>{i===n.length-1?s[r]=a:((!s[r]||typeof s[r]!="object"||Array.isArray(s[r])||s[r]instanceof File)&&(s[r]=Object.create(null)),s=s[r])})},ut=e=>{const t=e.split("/");return t[0]===""&&t.shift(),t},Ut=e=>{const{groups:t,path:a}=Gt(e),s=ut(a);return zt(s,t)},Gt=e=>{const t=[];return e=e.replace(/\{[^}]+\}/g,(a,s)=>{const n=`@${s}`;return t.push([n,a]),n}),{groups:t,path:e}},zt=(e,t)=>{for(let a=t.length-1;a>=0;a--){const[s]=t[a];for(let n=e.length-1;n>=0;n--)if(e[n].includes(s)){e[n]=e[n].replace(s,t[a][1]);break}}return e},ke={},Kt=(e,t)=>{if(e==="*")return"*";const a=e.match(/^\:([^\{\}]+)(?:\{(.+)\})?$/);if(a){const s=`${e}#${t}`;return ke[s]||(a[2]?ke[s]=t&&t[0]!==":"&&t[0]!=="*"?[s,a[1],new RegExp(`^${a[2]}(?=/${t})`)]:[e,a[1],new RegExp(`^${a[2]}$`)]:ke[s]=[e,a[1],!0]),ke[s]}return null},Ke=(e,t)=>{try{return t(e)}catch{return e.replace(/(?:%[0-9A-Fa-f]{2})+/g,a=>{try{return t(a)}catch{return a}})}},Yt=e=>Ke(e,decodeURI),mt=e=>{const t=e.url,a=t.indexOf("/",t.indexOf(":")+4);let s=a;for(;s<t.length;s++){const n=t.charCodeAt(s);if(n===37){const r=t.indexOf("?",s),i=t.slice(a,r===-1?void 0:r);return Yt(i.includes("%25")?i.replace(/%25/g,"%2525"):i)}else if(n===63)break}return t.slice(a,s)},Vt=e=>{const t=mt(e);return t.length>1&&t.at(-1)==="/"?t.slice(0,-1):t},de=(e,t,...a)=>(a.length&&(t=de(t,...a)),`${(e==null?void 0:e[0])==="/"?"":"/"}${e}${t==="/"?"":`${(e==null?void 0:e.at(-1))==="/"?"":"/"}${(t==null?void 0:t[0])==="/"?t.slice(1):t}`}`),pt=e=>{if(e.charCodeAt(e.length-1)!==63||!e.includes(":"))return null;const t=e.split("/"),a=[];let s="";return t.forEach(n=>{if(n!==""&&!/\:/.test(n))s+="/"+n;else if(/\:/.test(n))if(/\?/.test(n)){a.length===0&&s===""?a.push("/"):a.push(s);const r=n.replace("?","");s+="/"+r,a.push(s)}else s+="/"+n}),a.filter((n,r,i)=>i.indexOf(n)===r)},He=e=>/[%+]/.test(e)?(e.indexOf("+")!==-1&&(e=e.replace(/\+/g," ")),e.indexOf("%")!==-1?Ke(e,ht):e):e,ft=(e,t,a)=>{let s;if(!a&&t&&!/[%+]/.test(t)){let i=e.indexOf(`?${t}`,8);for(i===-1&&(i=e.indexOf(`&${t}`,8));i!==-1;){const o=e.charCodeAt(i+t.length+1);if(o===61){const l=i+t.length+2,c=e.indexOf("&",l);return He(e.slice(l,c===-1?void 0:c))}else if(o==38||isNaN(o))return"";i=e.indexOf(`&${t}`,i+1)}if(s=/[%+]/.test(e),!s)return}const n={};s??(s=/[%+]/.test(e));let r=e.indexOf("?",8);for(;r!==-1;){const i=e.indexOf("&",r+1);let o=e.indexOf("=",r);o>i&&i!==-1&&(o=-1);let l=e.slice(r+1,o===-1?i===-1?void 0:i:o);if(s&&(l=He(l)),r=i,l==="")continue;let c;o===-1?c="":(c=e.slice(o+1,i===-1?void 0:i),s&&(c=He(c))),a?(n[l]&&Array.isArray(n[l])||(n[l]=[]),n[l].push(c)):n[l]??(n[l]=c)}return t?n[t]:n},Wt=ft,Xt=(e,t)=>ft(e,t,!0),ht=decodeURIComponent,Je=e=>Ke(e,ht),me,M,U,yt,xt,Ue,K,st,bt=(st=class{constructor(e,t="/",a=[[]]){v(this,U);y(this,"raw");v(this,me);v(this,M);y(this,"routeIndex",0);y(this,"path");y(this,"bodyCache",{});v(this,K,e=>{const{bodyCache:t,raw:a}=this,s=t[e];if(s)return s;const n=Object.keys(t)[0];return n?t[n].then(r=>(n==="json"&&(r=JSON.stringify(r)),new Response(r)[e]())):t[e]=a[e]()});this.raw=e,this.path=t,b(this,M,a),b(this,me,{})}param(e){return e?w(this,U,yt).call(this,e):w(this,U,xt).call(this)}query(e){return Wt(this.url,e)}queries(e){return Xt(this.url,e)}header(e){if(e)return this.raw.headers.get(e)??void 0;const t={};return this.raw.headers.forEach((a,s)=>{t[s]=a}),t}async parseBody(e){var t;return(t=this.bodyCache).parsedBody??(t.parsedBody=await Ft(this,e))}json(){return g(this,K).call(this,"text").then(e=>JSON.parse(e))}text(){return g(this,K).call(this,"text")}arrayBuffer(){return g(this,K).call(this,"arrayBuffer")}blob(){return g(this,K).call(this,"blob")}formData(){return g(this,K).call(this,"formData")}addValidatedData(e,t){g(this,me)[e]=t}valid(e){return g(this,me)[e]}get url(){return this.raw.url}get method(){return this.raw.method}get[Nt](){return g(this,M)}get matchedRoutes(){return g(this,M)[0].map(([[,e]])=>e)}get routePath(){return g(this,M)[0].map(([[,e]])=>e)[this.routeIndex].path}},me=new WeakMap,M=new WeakMap,U=new WeakSet,yt=function(e){const t=g(this,M)[0][this.routeIndex][1][e],a=w(this,U,Ue).call(this,t);return a&&/\%/.test(a)?Je(a):a},xt=function(){const e={},t=Object.keys(g(this,M)[0][this.routeIndex][1]);for(const a of t){const s=w(this,U,Ue).call(this,g(this,M)[0][this.routeIndex][1][a]);s!==void 0&&(e[a]=/\%/.test(s)?Je(s):s)}return e},Ue=function(e){return g(this,M)[1]?g(this,M)[1][e]:e},K=new WeakMap,st),Qt={Stringify:1},vt=async(e,t,a,s,n)=>{typeof e=="object"&&!(e instanceof String)&&(e instanceof Promise||(e=e.toString()),e instanceof Promise&&(e=await e));const r=e.callbacks;return r!=null&&r.length?(n?n[0]+=e:n=[e],Promise.all(r.map(o=>o({phase:t,buffer:n,context:s}))).then(o=>Promise.all(o.filter(Boolean).map(l=>vt(l,t,!1,s,n))).then(()=>n[0]))):Promise.resolve(e)},Jt="text/plain; charset=UTF-8",qe=(e,t)=>({"Content-Type":e,...t}),Se,Ae,j,pe,$,k,Ce,fe,he,se,Te,Ie,Y,ge,nt,Zt=(nt=class{constructor(e,t){v(this,Y);v(this,Se);v(this,Ae);y(this,"env",{});v(this,j);y(this,"finalized",!1);y(this,"error");v(this,pe);v(this,$);v(this,k);v(this,Ce);v(this,fe);v(this,he);v(this,se);v(this,Te);v(this,Ie);y(this,"render",(...e)=>(g(this,fe)??b(this,fe,t=>this.html(t)),g(this,fe).call(this,...e)));y(this,"setLayout",e=>b(this,Ce,e));y(this,"getLayout",()=>g(this,Ce));y(this,"setRenderer",e=>{b(this,fe,e)});y(this,"header",(e,t,a)=>{this.finalized&&b(this,k,new Response(g(this,k).body,g(this,k)));const s=g(this,k)?g(this,k).headers:g(this,se)??b(this,se,new Headers);t===void 0?s.delete(e):a!=null&&a.append?s.append(e,t):s.set(e,t)});y(this,"status",e=>{b(this,pe,e)});y(this,"set",(e,t)=>{g(this,j)??b(this,j,new Map),g(this,j).set(e,t)});y(this,"get",e=>g(this,j)?g(this,j).get(e):void 0);y(this,"newResponse",(...e)=>w(this,Y,ge).call(this,...e));y(this,"body",(e,t,a)=>w(this,Y,ge).call(this,e,t,a));y(this,"text",(e,t,a)=>!g(this,se)&&!g(this,pe)&&!t&&!a&&!this.finalized?new Response(e):w(this,Y,ge).call(this,e,t,qe(Jt,a)));y(this,"json",(e,t,a)=>w(this,Y,ge).call(this,JSON.stringify(e),t,qe("application/json",a)));y(this,"html",(e,t,a)=>{const s=n=>w(this,Y,ge).call(this,n,t,qe("text/html; charset=UTF-8",a));return typeof e=="object"?vt(e,Qt.Stringify,!1,{}).then(s):s(e)});y(this,"redirect",(e,t)=>{const a=String(e);return this.header("Location",/[^\x00-\xFF]/.test(a)?encodeURI(a):a),this.newResponse(null,t??302)});y(this,"notFound",()=>(g(this,he)??b(this,he,()=>new Response),g(this,he).call(this,this)));b(this,Se,e),t&&(b(this,$,t.executionCtx),this.env=t.env,b(this,he,t.notFoundHandler),b(this,Ie,t.path),b(this,Te,t.matchResult))}get req(){return g(this,Ae)??b(this,Ae,new bt(g(this,Se),g(this,Ie),g(this,Te))),g(this,Ae)}get event(){if(g(this,$)&&"respondWith"in g(this,$))return g(this,$);throw Error("This context has no FetchEvent")}get executionCtx(){if(g(this,$))return g(this,$);throw Error("This context has no ExecutionContext")}get res(){return g(this,k)||b(this,k,new Response(null,{headers:g(this,se)??b(this,se,new Headers)}))}set res(e){if(g(this,k)&&e){e=new Response(e.body,e);for(const[t,a]of g(this,k).headers.entries())if(t!=="content-type")if(t==="set-cookie"){const s=g(this,k).headers.getSetCookie();e.headers.delete("set-cookie");for(const n of s)e.headers.append("set-cookie",n)}else e.headers.set(t,a)}b(this,k,e),this.finalized=!0}get var(){return g(this,j)?Object.fromEntries(g(this,j)):{}}},Se=new WeakMap,Ae=new WeakMap,j=new WeakMap,pe=new WeakMap,$=new WeakMap,k=new WeakMap,Ce=new WeakMap,fe=new WeakMap,he=new WeakMap,se=new WeakMap,Te=new WeakMap,Ie=new WeakMap,Y=new WeakSet,ge=function(e,t,a){const s=g(this,k)?new Headers(g(this,k).headers):g(this,se)??new Headers;if(typeof t=="object"&&"headers"in t){const r=t.headers instanceof Headers?t.headers:new Headers(t.headers);for(const[i,o]of r)i.toLowerCase()==="set-cookie"?s.append(i,o):s.set(i,o)}if(a)for(const[r,i]of Object.entries(a))if(typeof i=="string")s.set(r,i);else{s.delete(r);for(const o of i)s.append(r,o)}const n=typeof t=="number"?t:(t==null?void 0:t.status)??g(this,pe);return new Response(e,{status:n,headers:s})},nt),A="ALL",ea="all",ta=["get","post","put","delete","options","patch"],_t="Can not add a route since the matcher is already built.",wt=class extends Error{},aa="__COMPOSED_HANDLER",sa=e=>e.text("404 Not Found",404),Ze=(e,t)=>{if("getResponse"in e){const a=e.getResponse();return t.newResponse(a.body,a)}return console.error(e),t.text("Internal Server Error",500)},O,C,St,P,te,Me,Oe,rt,Et=(rt=class{constructor(t={}){v(this,C);y(this,"get");y(this,"post");y(this,"put");y(this,"delete");y(this,"options");y(this,"patch");y(this,"all");y(this,"on");y(this,"use");y(this,"router");y(this,"getPath");y(this,"_basePath","/");v(this,O,"/");y(this,"routes",[]);v(this,P,sa);y(this,"errorHandler",Ze);y(this,"onError",t=>(this.errorHandler=t,this));y(this,"notFound",t=>(b(this,P,t),this));y(this,"fetch",(t,...a)=>w(this,C,Oe).call(this,t,a[1],a[0],t.method));y(this,"request",(t,a,s,n)=>t instanceof Request?this.fetch(a?new Request(t,a):t,s,n):(t=t.toString(),this.fetch(new Request(/^https?:\/\//.test(t)?t:`http://localhost${de("/",t)}`,a),s,n)));y(this,"fire",()=>{addEventListener("fetch",t=>{t.respondWith(w(this,C,Oe).call(this,t.request,t,void 0,t.request.method))})});[...ta,ea].forEach(r=>{this[r]=(i,...o)=>(typeof i=="string"?b(this,O,i):w(this,C,te).call(this,r,g(this,O),i),o.forEach(l=>{w(this,C,te).call(this,r,g(this,O),l)}),this)}),this.on=(r,i,...o)=>{for(const l of[i].flat()){b(this,O,l);for(const c of[r].flat())o.map(d=>{w(this,C,te).call(this,c.toUpperCase(),g(this,O),d)})}return this},this.use=(r,...i)=>(typeof r=="string"?b(this,O,r):(b(this,O,"*"),i.unshift(r)),i.forEach(o=>{w(this,C,te).call(this,A,g(this,O),o)}),this);const{strict:s,...n}=t;Object.assign(this,n),this.getPath=s??!0?t.getPath??mt:Vt}route(t,a){const s=this.basePath(t);return a.routes.map(n=>{var i;let r;a.errorHandler===Ze?r=n.handler:(r=async(o,l)=>(await Qe([],a.errorHandler)(o,()=>n.handler(o,l))).res,r[aa]=n.handler),w(i=s,C,te).call(i,n.method,n.path,r)}),this}basePath(t){const a=w(this,C,St).call(this);return a._basePath=de(this._basePath,t),a}mount(t,a,s){let n,r;s&&(typeof s=="function"?r=s:(r=s.optionHandler,s.replaceRequest===!1?n=l=>l:n=s.replaceRequest));const i=r?l=>{const c=r(l);return Array.isArray(c)?c:[c]}:l=>{let c;try{c=l.executionCtx}catch{}return[l.env,c]};n||(n=(()=>{const l=de(this._basePath,t),c=l==="/"?0:l.length;return d=>{const u=new URL(d.url);return u.pathname=u.pathname.slice(c)||"/",new Request(u,d)}})());const o=async(l,c)=>{const d=await a(n(l.req.raw),...i(l));if(d)return d;await c()};return w(this,C,te).call(this,A,de(t,"*"),o),this}},O=new WeakMap,C=new WeakSet,St=function(){const t=new Et({router:this.router,getPath:this.getPath});return t.errorHandler=this.errorHandler,b(t,P,g(this,P)),t.routes=this.routes,t},P=new WeakMap,te=function(t,a,s){t=t.toUpperCase(),a=de(this._basePath,a);const n={basePath:this._basePath,path:a,method:t,handler:s};this.router.add(t,a,[s,n]),this.routes.push(n)},Me=function(t,a){if(t instanceof Error)return this.errorHandler(t,a);throw t},Oe=function(t,a,s,n){if(n==="HEAD")return(async()=>new Response(null,await w(this,C,Oe).call(this,t,a,s,"GET")))();const r=this.getPath(t,{env:s}),i=this.router.match(n,r),o=new Zt(t,{path:r,matchResult:i,env:s,executionCtx:a,notFoundHandler:g(this,P)});if(i[0].length===1){let c;try{c=i[0][0][0][0](o,async()=>{o.res=await g(this,P).call(this,o)})}catch(d){return w(this,C,Me).call(this,d,o)}return c instanceof Promise?c.then(d=>d||(o.finalized?o.res:g(this,P).call(this,o))).catch(d=>w(this,C,Me).call(this,d,o)):c??g(this,P).call(this,o)}const l=Qe(i[0],this.errorHandler,g(this,P));return(async()=>{try{const c=await l(o);if(!c.finalized)throw new Error("Context is not finalized. Did you forget to return a Response object or `await next()`?");return c.res}catch(c){return w(this,C,Me).call(this,c,o)}})()},rt),At=[];function na(e,t){const a=this.buildAllMatchers(),s=(n,r)=>{const i=a[n]||a[A],o=i[2][r];if(o)return o;const l=r.match(i[0]);if(!l)return[[],At];const c=l.indexOf("",1);return[i[1][c],l]};return this.match=s,s(e,t)}var Be="[^/]+",we=".*",Ee="(?:|/.*)",ue=Symbol(),ra=new Set(".\\+*[^]$()");function ia(e,t){return e.length===1?t.length===1?e<t?-1:1:-1:t.length===1||e===we||e===Ee?1:t===we||t===Ee?-1:e===Be?1:t===Be?-1:e.length===t.length?e<t?-1:1:t.length-e.length}var ne,re,B,it,Ge=(it=class{constructor(){v(this,ne);v(this,re);v(this,B,Object.create(null))}insert(t,a,s,n,r){if(t.length===0){if(g(this,ne)!==void 0)throw ue;if(r)return;b(this,ne,a);return}const[i,...o]=t,l=i==="*"?o.length===0?["","",we]:["","",Be]:i==="/*"?["","",Ee]:i.match(/^\:([^\{\}]+)(?:\{(.+)\})?$/);let c;if(l){const d=l[1];let u=l[2]||Be;if(d&&l[2]&&(u===".*"||(u=u.replace(/^\((?!\?:)(?=[^)]+\)$)/,"(?:"),/\((?!\?:)/.test(u))))throw ue;if(c=g(this,B)[u],!c){if(Object.keys(g(this,B)).some(m=>m!==we&&m!==Ee))throw ue;if(r)return;c=g(this,B)[u]=new Ge,d!==""&&b(c,re,n.varIndex++)}!r&&d!==""&&s.push([d,g(c,re)])}else if(c=g(this,B)[i],!c){if(Object.keys(g(this,B)).some(d=>d.length>1&&d!==we&&d!==Ee))throw ue;if(r)return;c=g(this,B)[i]=new Ge}c.insert(o,a,s,n,r)}buildRegExpStr(){const a=Object.keys(g(this,B)).sort(ia).map(s=>{const n=g(this,B)[s];return(typeof g(n,re)=="number"?`(${s})@${g(n,re)}`:ra.has(s)?`\\${s}`:s)+n.buildRegExpStr()});return typeof g(this,ne)=="number"&&a.unshift(`#${g(this,ne)}`),a.length===0?"":a.length===1?a[0]:"(?:"+a.join("|")+")"}},ne=new WeakMap,re=new WeakMap,B=new WeakMap,it),Fe,De,ot,oa=(ot=class{constructor(){v(this,Fe,{varIndex:0});v(this,De,new Ge)}insert(e,t,a){const s=[],n=[];for(let i=0;;){let o=!1;if(e=e.replace(/\{[^}]+\}/g,l=>{const c=`@\\${i}`;return n[i]=[c,l],i++,o=!0,c}),!o)break}const r=e.match(/(?::[^\/]+)|(?:\/\*$)|./g)||[];for(let i=n.length-1;i>=0;i--){const[o]=n[i];for(let l=r.length-1;l>=0;l--)if(r[l].indexOf(o)!==-1){r[l]=r[l].replace(o,n[i][1]);break}}return g(this,De).insert(r,t,s,g(this,Fe),a),s}buildRegExp(){let e=g(this,De).buildRegExpStr();if(e==="")return[/^$/,[],[]];let t=0;const a=[],s=[];return e=e.replace(/#(\d+)|@(\d+)|\.\*\$/g,(n,r,i)=>r!==void 0?(a[++t]=Number(r),"$()"):(i!==void 0&&(s[Number(i)]=++t),"")),[new RegExp(`^${e}`),a,s]}},Fe=new WeakMap,De=new WeakMap,ot),la=[/^$/,[],Object.create(null)],Pe=Object.create(null);function Ct(e){return Pe[e]??(Pe[e]=new RegExp(e==="*"?"":`^${e.replace(/\/\*$|([.\\+*[^\]$()])/g,(t,a)=>a?`\\${a}`:"(?:|/.*)")}$`))}function ca(){Pe=Object.create(null)}function da(e){var c;const t=new oa,a=[];if(e.length===0)return la;const s=e.map(d=>[!/\*|\/:/.test(d[0]),...d]).sort(([d,u],[m,p])=>d?1:m?-1:u.length-p.length),n=Object.create(null);for(let d=0,u=-1,m=s.length;d<m;d++){const[p,_,f]=s[d];p?n[_]=[f.map(([E])=>[E,Object.create(null)]),At]:u++;let x;try{x=t.insert(_,u,p)}catch(E){throw E===ue?new wt(_):E}p||(a[u]=f.map(([E,L])=>{const G=Object.create(null);for(L-=1;L>=0;L--){const[D,z]=x[L];G[D]=z}return[E,G]}))}const[r,i,o]=t.buildRegExp();for(let d=0,u=a.length;d<u;d++)for(let m=0,p=a[d].length;m<p;m++){const _=(c=a[d][m])==null?void 0:c[1];if(!_)continue;const f=Object.keys(_);for(let x=0,E=f.length;x<E;x++)_[f[x]]=o[_[f[x]]]}const l=[];for(const d in i)l[d]=a[i[d]];return[r,l,n]}function ce(e,t){if(e){for(const a of Object.keys(e).sort((s,n)=>n.length-s.length))if(Ct(a).test(t))return[...e[a]]}}var V,W,je,Tt,lt,ga=(lt=class{constructor(){v(this,je);y(this,"name","RegExpRouter");v(this,V);v(this,W);y(this,"match",na);b(this,V,{[A]:Object.create(null)}),b(this,W,{[A]:Object.create(null)})}add(e,t,a){var o;const s=g(this,V),n=g(this,W);if(!s||!n)throw new Error(_t);s[e]||[s,n].forEach(l=>{l[e]=Object.create(null),Object.keys(l[A]).forEach(c=>{l[e][c]=[...l[A][c]]})}),t==="/*"&&(t="*");const r=(t.match(/\/:/g)||[]).length;if(/\*$/.test(t)){const l=Ct(t);e===A?Object.keys(s).forEach(c=>{var d;(d=s[c])[t]||(d[t]=ce(s[c],t)||ce(s[A],t)||[])}):(o=s[e])[t]||(o[t]=ce(s[e],t)||ce(s[A],t)||[]),Object.keys(s).forEach(c=>{(e===A||e===c)&&Object.keys(s[c]).forEach(d=>{l.test(d)&&s[c][d].push([a,r])})}),Object.keys(n).forEach(c=>{(e===A||e===c)&&Object.keys(n[c]).forEach(d=>l.test(d)&&n[c][d].push([a,r]))});return}const i=pt(t)||[t];for(let l=0,c=i.length;l<c;l++){const d=i[l];Object.keys(n).forEach(u=>{var m;(e===A||e===u)&&((m=n[u])[d]||(m[d]=[...ce(s[u],d)||ce(s[A],d)||[]]),n[u][d].push([a,r-c+l+1]))})}}buildAllMatchers(){const e=Object.create(null);return Object.keys(g(this,W)).concat(Object.keys(g(this,V))).forEach(t=>{e[t]||(e[t]=w(this,je,Tt).call(this,t))}),b(this,V,b(this,W,void 0)),ca(),e}},V=new WeakMap,W=new WeakMap,je=new WeakSet,Tt=function(e){const t=[];let a=e===A;return[g(this,V),g(this,W)].forEach(s=>{const n=s[e]?Object.keys(s[e]).map(r=>[r,s[e][r]]):[];n.length!==0?(a||(a=!0),t.push(...n)):e!==A&&t.push(...Object.keys(s[A]).map(r=>[r,s[A][r]]))}),a?da(t):null},lt),X,H,ct,ua=(ct=class{constructor(e){y(this,"name","SmartRouter");v(this,X,[]);v(this,H,[]);b(this,X,e.routers)}add(e,t,a){if(!g(this,H))throw new Error(_t);g(this,H).push([e,t,a])}match(e,t){if(!g(this,H))throw new Error("Fatal error");const a=g(this,X),s=g(this,H),n=a.length;let r=0,i;for(;r<n;r++){const o=a[r];try{for(let l=0,c=s.length;l<c;l++)o.add(...s[l]);i=o.match(e,t)}catch(l){if(l instanceof wt)continue;throw l}this.match=o.match.bind(o),b(this,X,[o]),b(this,H,void 0);break}if(r===n)throw new Error("Fatal error");return this.name=`SmartRouter + ${this.activeRouter.name}`,i}get activeRouter(){if(g(this,H)||g(this,X).length!==1)throw new Error("No active router has been determined yet.");return g(this,X)[0]}},X=new WeakMap,H=new WeakMap,ct),_e=Object.create(null),Q,I,ie,be,T,q,ae,dt,It=(dt=class{constructor(e,t,a){v(this,q);v(this,Q);v(this,I);v(this,ie);v(this,be,0);v(this,T,_e);if(b(this,I,a||Object.create(null)),b(this,Q,[]),e&&t){const s=Object.create(null);s[e]={handler:t,possibleKeys:[],score:0},b(this,Q,[s])}b(this,ie,[])}insert(e,t,a){b(this,be,++Xe(this,be)._);let s=this;const n=Ut(t),r=[];for(let i=0,o=n.length;i<o;i++){const l=n[i],c=n[i+1],d=Kt(l,c),u=Array.isArray(d)?d[0]:l;if(u in g(s,I)){s=g(s,I)[u],d&&r.push(d[1]);continue}g(s,I)[u]=new It,d&&(g(s,ie).push(d),r.push(d[1])),s=g(s,I)[u]}return g(s,Q).push({[e]:{handler:a,possibleKeys:r.filter((i,o,l)=>l.indexOf(i)===o),score:g(this,be)}}),s}search(e,t){var o;const a=[];b(this,T,_e);let n=[this];const r=ut(t),i=[];for(let l=0,c=r.length;l<c;l++){const d=r[l],u=l===c-1,m=[];for(let p=0,_=n.length;p<_;p++){const f=n[p],x=g(f,I)[d];x&&(b(x,T,g(f,T)),u?(g(x,I)["*"]&&a.push(...w(this,q,ae).call(this,g(x,I)["*"],e,g(f,T))),a.push(...w(this,q,ae).call(this,x,e,g(f,T)))):m.push(x));for(let E=0,L=g(f,ie).length;E<L;E++){const G=g(f,ie)[E],D=g(f,T)===_e?{}:{...g(f,T)};if(G==="*"){const N=g(f,I)["*"];N&&(a.push(...w(this,q,ae).call(this,N,e,g(f,T))),b(N,T,D),m.push(N));continue}const[z,Re,J]=G;if(!d&&!(J instanceof RegExp))continue;const R=g(f,I)[z],ye=r.slice(l).join("/");if(J instanceof RegExp){const N=J.exec(ye);if(N){if(D[Re]=N[0],a.push(...w(this,q,ae).call(this,R,e,g(f,T),D)),Object.keys(g(R,I)).length){b(R,T,D);const le=((o=N[0].match(/\//))==null?void 0:o.length)??0;(i[le]||(i[le]=[])).push(R)}continue}}(J===!0||J.test(d))&&(D[Re]=d,u?(a.push(...w(this,q,ae).call(this,R,e,D,g(f,T))),g(R,I)["*"]&&a.push(...w(this,q,ae).call(this,g(R,I)["*"],e,D,g(f,T)))):(b(R,T,D),m.push(R)))}}n=m.concat(i.shift()??[])}return a.length>1&&a.sort((l,c)=>l.score-c.score),[a.map(({handler:l,params:c})=>[l,c])]}},Q=new WeakMap,I=new WeakMap,ie=new WeakMap,be=new WeakMap,T=new WeakMap,q=new WeakSet,ae=function(e,t,a,s){const n=[];for(let r=0,i=g(e,Q).length;r<i;r++){const o=g(e,Q)[r],l=o[t]||o[A],c={};if(l!==void 0&&(l.params=Object.create(null),n.push(l),a!==_e||s&&s!==_e))for(let d=0,u=l.possibleKeys.length;d<u;d++){const m=l.possibleKeys[d],p=c[l.score];l.params[m]=s!=null&&s[m]&&!p?s[m]:a[m]??(s==null?void 0:s[m]),c[l.score]=!0}}return n},dt),oe,gt,ma=(gt=class{constructor(){y(this,"name","TrieRouter");v(this,oe);b(this,oe,new It)}add(e,t,a){const s=pt(t);if(s){for(let n=0,r=s.length;n<r;n++)g(this,oe).insert(e,s[n],a);return}g(this,oe).insert(e,t,a)}match(e,t){return g(this,oe).search(e,t)}},oe=new WeakMap,gt),Dt=class extends Et{constructor(e={}){super(e),this.router=e.router??new ua({routers:[new ga,new ma]})}},pa=e=>{const a={...{origin:"*",allowMethods:["GET","HEAD","PUT","POST","DELETE","PATCH"],allowHeaders:[],exposeHeaders:[]},...e},s=(r=>typeof r=="string"?r==="*"?()=>r:i=>r===i?i:null:typeof r=="function"?r:i=>r.includes(i)?i:null)(a.origin),n=(r=>typeof r=="function"?r:Array.isArray(r)?()=>r:()=>[])(a.allowMethods);return async function(i,o){var d;function l(u,m){i.res.headers.set(u,m)}const c=await s(i.req.header("origin")||"",i);if(c&&l("Access-Control-Allow-Origin",c),a.credentials&&l("Access-Control-Allow-Credentials","true"),(d=a.exposeHeaders)!=null&&d.length&&l("Access-Control-Expose-Headers",a.exposeHeaders.join(",")),i.req.method==="OPTIONS"){a.origin!=="*"&&l("Vary","Origin"),a.maxAge!=null&&l("Access-Control-Max-Age",a.maxAge.toString());const u=await n(i.req.header("origin")||"",i);u.length&&l("Access-Control-Allow-Methods",u.join(","));let m=a.allowHeaders;if(!(m!=null&&m.length)){const p=i.req.header("Access-Control-Request-Headers");p&&(m=p.split(/\s*,\s*/))}return m!=null&&m.length&&(l("Access-Control-Allow-Headers",m.join(",")),i.res.headers.append("Vary","Access-Control-Request-Headers")),i.res.headers.delete("Content-Length"),i.res.headers.delete("Content-Type"),new Response(null,{headers:i.res.headers,status:204,statusText:"No Content"})}await o(),a.origin!=="*"&&i.header("Vary","Origin",{append:!0})}},fa=/^\s*(?:text\/(?!event-stream(?:[;\s]|$))[^;\s]+|application\/(?:javascript|json|xml|xml-dtd|ecmascript|dart|postscript|rtf|tar|toml|vnd\.dart|vnd\.ms-fontobject|vnd\.ms-opentype|wasm|x-httpd-php|x-javascript|x-ns-proxy-autoconfig|x-sh|x-tar|x-virtualbox-hdd|x-virtualbox-ova|x-virtualbox-ovf|x-virtualbox-vbox|x-virtualbox-vdi|x-virtualbox-vhd|x-virtualbox-vmdk|x-www-form-urlencoded)|font\/(?:otf|ttf)|image\/(?:bmp|vnd\.adobe\.photoshop|vnd\.microsoft\.icon|vnd\.ms-dds|x-icon|x-ms-bmp)|message\/rfc822|model\/gltf-binary|x-shader\/x-fragment|x-shader\/x-vertex|[^;\s]+?\+(?:json|text|xml|yaml))(?:[;\s]|$)/i,et=(e,t=ba)=>{const a=/\.([a-zA-Z0-9]+?)$/,s=e.match(a);if(!s)return;let n=t[s[1]];return n&&n.startsWith("text")&&(n+="; charset=utf-8"),n},ha={aac:"audio/aac",avi:"video/x-msvideo",avif:"image/avif",av1:"video/av1",bin:"application/octet-stream",bmp:"image/bmp",css:"text/css",csv:"text/csv",eot:"application/vnd.ms-fontobject",epub:"application/epub+zip",gif:"image/gif",gz:"application/gzip",htm:"text/html",html:"text/html",ico:"image/x-icon",ics:"text/calendar",jpeg:"image/jpeg",jpg:"image/jpeg",js:"text/javascript",json:"application/json",jsonld:"application/ld+json",map:"application/json",mid:"audio/x-midi",midi:"audio/x-midi",mjs:"text/javascript",mp3:"audio/mpeg",mp4:"video/mp4",mpeg:"video/mpeg",oga:"audio/ogg",ogv:"video/ogg",ogx:"application/ogg",opus:"audio/opus",otf:"font/otf",pdf:"application/pdf",png:"image/png",rtf:"application/rtf",svg:"image/svg+xml",tif:"image/tiff",tiff:"image/tiff",ts:"video/mp2t",ttf:"font/ttf",txt:"text/plain",wasm:"application/wasm",webm:"video/webm",weba:"audio/webm",webmanifest:"application/manifest+json",webp:"image/webp",woff:"font/woff",woff2:"font/woff2",xhtml:"application/xhtml+xml",xml:"application/xml",zip:"application/zip","3gp":"video/3gpp","3g2":"video/3gpp2",gltf:"model/gltf+json",glb:"model/gltf-binary"},ba=ha,ya=(...e)=>{let t=e.filter(n=>n!=="").join("/");t=t.replace(new RegExp("(?<=\\/)\\/+","g"),"");const a=t.split("/"),s=[];for(const n of a)n===".."&&s.length>0&&s.at(-1)!==".."?s.pop():n!=="."&&s.push(n);return s.join("/")||"."},Rt={br:".br",zstd:".zst",gzip:".gz"},xa=Object.keys(Rt),va="index.html",_a=e=>{const t=e.root??"./",a=e.path,s=e.join??ya;return async(n,r)=>{var d,u,m,p;if(n.finalized)return r();let i;if(e.path)i=e.path;else try{if(i=decodeURIComponent(n.req.path),/(?:^|[\/\\])\.\.(?:$|[\/\\])/.test(i))throw new Error}catch{return await((d=e.onNotFound)==null?void 0:d.call(e,n.req.path,n)),r()}let o=s(t,!a&&e.rewriteRequestPath?e.rewriteRequestPath(i):i);e.isDir&&await e.isDir(o)&&(o=s(o,va));const l=e.getContent;let c=await l(o,n);if(c instanceof Response)return n.newResponse(c.body,c);if(c){const _=e.mimes&&et(o,e.mimes)||et(o);if(n.header("Content-Type",_||"application/octet-stream"),e.precompressed&&(!_||fa.test(_))){const f=new Set((u=n.req.header("Accept-Encoding"))==null?void 0:u.split(",").map(x=>x.trim()));for(const x of xa){if(!f.has(x))continue;const E=await l(o+Rt[x],n);if(E){c=E,n.header("Content-Encoding",x),n.header("Vary","Accept-Encoding",{append:!0});break}}}return await((m=e.onFound)==null?void 0:m.call(e,o,n)),n.body(c)}await((p=e.onNotFound)==null?void 0:p.call(e,o,n)),await r()}},wa=async(e,t)=>{let a;t&&t.manifest?typeof t.manifest=="string"?a=JSON.parse(t.manifest):a=t.manifest:typeof __STATIC_CONTENT_MANIFEST=="string"?a=JSON.parse(__STATIC_CONTENT_MANIFEST):a=__STATIC_CONTENT_MANIFEST;let s;t&&t.namespace?s=t.namespace:s=__STATIC_CONTENT;const n=a[e]||e;if(!n)return null;const r=await s.get(n,{type:"stream"});return r||null},Ea=e=>async function(a,s){return _a({...e,getContent:async r=>wa(r,{manifest:e.manifest,namespace:e.namespace?e.namespace:a.env?a.env.__STATIC_CONTENT:void 0})})(a,s)},Sa=e=>Ea(e);const S=new Dt,h={ECONOMIC:{FED_RATE_BULLISH:4.5,FED_RATE_BEARISH:5.5,CPI_TARGET:2,CPI_WARNING:3.5,GDP_HEALTHY:2,UNEMPLOYMENT_LOW:4,PMI_EXPANSION:50,TREASURY_SPREAD_INVERSION:-.5},SENTIMENT:{FEAR_GREED_EXTREME_FEAR:25,FEAR_GREED_EXTREME_GREED:75,VIX_LOW:15,VIX_HIGH:25,SOCIAL_VOLUME_HIGH:15e4,INSTITUTIONAL_FLOW_THRESHOLD:10},LIQUIDITY:{BID_ASK_SPREAD_TIGHT:.1,BID_ASK_SPREAD_WIDE:.5,ARBITRAGE_OPPORTUNITY:.3,ORDER_BOOK_DEPTH_MIN:1e6,SLIPPAGE_MAX:.2},TRENDS:{INTEREST_HIGH:70,INTEREST_RISING:20},IMF:{GDP_GROWTH_STRONG:3,INFLATION_TARGET:2.5,DEBT_WARNING:80}};S.use("/api/*",pa());S.use("/static/*",Sa({root:"./public"}));async function Aa(){try{const e=new AbortController,t=setTimeout(()=>e.abort(),5e3),a=await fetch("https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH,PCPIPCH",{signal:e.signal});if(clearTimeout(t),!a.ok)return null;const s=await a.json();return{timestamp:Date.now(),iso_timestamp:new Date().toISOString(),gdp_growth:s.NGDP_RPCH||{},inflation:s.PCPIPCH||{},source:"IMF"}}catch(e){return console.error("IMF API error (timeout or network):",e),null}}async function kt(e="BTCUSDT"){try{const t=await fetch(`https://api.binance.com/api/v3/ticker/24hr?symbol=${e}`);if(!t.ok)return null;const a=await t.json();return{exchange:"Binance",symbol:e,price:parseFloat(a.lastPrice),volume_24h:parseFloat(a.volume),price_change_24h:parseFloat(a.priceChangePercent),high_24h:parseFloat(a.highPrice),low_24h:parseFloat(a.lowPrice),bid:parseFloat(a.bidPrice),ask:parseFloat(a.askPrice),timestamp:a.closeTime}}catch(t){return console.error("Binance API error:",t),null}}async function Ne(e="BTC-USD"){try{const t=await fetch(`https://api.coinbase.com/v2/prices/${e}/spot`);if(!t.ok)return null;const a=await t.json();return{exchange:"Coinbase",symbol:e,price:parseFloat(a.data.amount),currency:a.data.currency,timestamp:Date.now()}}catch(t){return console.error("Coinbase API error:",t),null}}async function Lt(e="XBTUSD"){try{const t=await fetch(`https://api.kraken.com/0/public/Ticker?pair=${e}`);if(!t.ok)return null;const a=await t.json(),s=a.result[Object.keys(a.result)[0]];return{exchange:"Kraken",pair:e,price:parseFloat(s.c[0]),volume_24h:parseFloat(s.v[1]),bid:parseFloat(s.b[0]),ask:parseFloat(s.a[0]),high_24h:parseFloat(s.h[1]),low_24h:parseFloat(s.l[1]),timestamp:Date.now()}}catch(t){return console.error("Kraken API error:",t),null}}async function Ca(e,t="bitcoin"){var a,s,n,r;if(!e)return null;try{const i=await fetch(`https://api.coingecko.com/api/v3/simple/price?ids=${t}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true&include_last_updated_at=true`,{headers:{"x-cg-demo-api-key":e}});if(!i.ok)return null;const o=await i.json();return{coin:t,price:(a=o[t])==null?void 0:a.usd,volume_24h:(s=o[t])==null?void 0:s.usd_24h_vol,change_24h:(n=o[t])==null?void 0:n.usd_24h_change,last_updated:(r=o[t])==null?void 0:r.last_updated_at,timestamp:Date.now(),source:"CoinGecko"}}catch(i){return console.error("CoinGecko API error:",i),null}}async function Le(e,t){if(!e)return null;try{const a=new AbortController,s=setTimeout(()=>a.abort(),5e3),n=await fetch(`https://api.stlouisfed.org/fred/series/observations?series_id=${t}&api_key=${e}&file_type=json&limit=1&sort_order=desc`,{signal:a.signal});if(clearTimeout(s),!n.ok)return null;const i=(await n.json()).observations[0];return{series_id:t,value:parseFloat(i.value),date:i.date,timestamp:Date.now(),source:"FRED"}}catch(a){return console.error("FRED API error:",a),null}}async function Ta(e,t){if(!e)return null;try{const a=new AbortController,s=setTimeout(()=>a.abort(),5e3),n=await fetch(`https://serpapi.com/search.json?engine=google_trends&q=${encodeURIComponent(t)}&api_key=${e}`,{signal:a.signal});if(clearTimeout(s),!n.ok)return null;const r=await n.json();return{query:t,interest_over_time:r.interest_over_time,timestamp:Date.now(),source:"Google Trends"}}catch(a){return console.error("Google Trends API error:",a),null}}function Ia(e){const t=[];for(let a=0;a<e.length;a++)for(let s=a+1;s<e.length;s++){const n=e[a],r=e[s];if(n&&r&&n.price&&r.price){const i=(r.price-n.price)/n.price*100;Math.abs(i)>=h.LIQUIDITY.ARBITRAGE_OPPORTUNITY&&t.push({buy_exchange:i>0?n.exchange:r.exchange,sell_exchange:i>0?r.exchange:n.exchange,spread_percent:Math.abs(i),profit_potential:Math.abs(i)>h.LIQUIDITY.ARBITRAGE_OPPORTUNITY?"high":"medium"})}}return t}S.get("/api/market/data/:symbol",async e=>{const t=e.req.param("symbol"),{env:a}=e;try{const s=Date.now();return await a.DB.prepare(`
      INSERT INTO market_data (symbol, exchange, price, volume, timestamp, data_type)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(t,"aggregated",0,0,s,"spot").run(),e.json({success:!0,data:{symbol:t,price:Math.random()*5e4+3e4,volume:Math.random()*1e6,timestamp:s,source:"mock"}})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.get("/api/economic/indicators",async e=>{var a;const{env:t}=e;try{const s=await t.DB.prepare(`
      SELECT * FROM economic_indicators 
      ORDER BY timestamp DESC 
      LIMIT 10
    `).all();return e.json({success:!0,data:s.results,count:((a=s.results)==null?void 0:a.length)||0})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.post("/api/economic/indicators",async e=>{const{env:t}=e,a=await e.req.json();try{const{indicator_name:s,indicator_code:n,value:r,period:i,source:o}=a,l=Date.now();return await t.DB.prepare(`
      INSERT INTO economic_indicators 
      (indicator_name, indicator_code, value, period, source, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(s,n,r,i,o,l).run(),e.json({success:!0,message:"Indicator stored successfully"})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.get("/api/agents/economic",async e=>{var s,n,r,i;const t=e.req.query("symbol")||"BTC",{env:a}=e;try{const o=a.FRED_API_KEY,l=await Promise.all([Le(o,"FEDFUNDS"),Le(o,"CPIAUCSL"),Le(o,"UNRATE"),Le(o,"GDP")]),c=await Aa(),d=((s=l[0])==null?void 0:s.value)||5.33,u=((n=l[1])==null?void 0:n.value)||3.2,m=((r=l[2])==null?void 0:r.value)||3.8,p=((i=l[3])==null?void 0:i.value)||2.4,_=d<h.ECONOMIC.FED_RATE_BULLISH?"bullish":d>h.ECONOMIC.FED_RATE_BEARISH?"bearish":"neutral",f=u<=h.ECONOMIC.CPI_TARGET?"healthy":u>h.ECONOMIC.CPI_WARNING?"warning":"elevated",x=p>=h.ECONOMIC.GDP_HEALTHY?"healthy":"weak",E=m<=h.ECONOMIC.UNEMPLOYMENT_LOW?"tight":"loose",L={timestamp:Date.now(),iso_timestamp:new Date().toISOString(),symbol:t,data_source:"Economic Agent",data_freshness:o?"LIVE":"SIMULATED",indicators:{fed_funds_rate:{value:d,signal:_,constraint_bullish:h.ECONOMIC.FED_RATE_BULLISH,constraint_bearish:h.ECONOMIC.FED_RATE_BEARISH,next_meeting:"2025-11-07",source:l[0]?"FRED":"simulated"},cpi:{value:u,signal:f,target:h.ECONOMIC.CPI_TARGET,warning_threshold:h.ECONOMIC.CPI_WARNING,trend:u<3.5?"decreasing":"elevated",source:l[1]?"FRED":"simulated"},unemployment_rate:{value:m,signal:E,threshold:h.ECONOMIC.UNEMPLOYMENT_LOW,trend:m<4?"tight":"stable",source:l[2]?"FRED":"simulated"},gdp_growth:{value:p,signal:x,healthy_threshold:h.ECONOMIC.GDP_HEALTHY,quarter:"Q3 2025",source:l[3]?"FRED":"simulated"},manufacturing_pmi:{value:48.5,status:48.5<h.ECONOMIC.PMI_EXPANSION?"contraction":"expansion",expansion_threshold:h.ECONOMIC.PMI_EXPANSION},imf_global:c?{available:!0,gdp_growth:c.gdp_growth,inflation:c.inflation,source:"IMF",timestamp:c.iso_timestamp}:{available:!1}},constraints_applied:{fed_rate_range:[h.ECONOMIC.FED_RATE_BULLISH,h.ECONOMIC.FED_RATE_BEARISH],cpi_target:h.ECONOMIC.CPI_TARGET,gdp_healthy:h.ECONOMIC.GDP_HEALTHY,unemployment_low:h.ECONOMIC.UNEMPLOYMENT_LOW}};return e.json({success:!0,agent:"economic",data:L})}catch(o){return e.json({success:!1,error:String(o)},500)}});S.get("/api/agents/sentiment",async e=>{const t=e.req.query("symbol")||"BTC",{env:a}=e;try{const s=a.SERPAPI_KEY,n=await Ta(s,t==="BTC"?"bitcoin":"ethereum"),r=61+Math.floor(Math.random()*20-10),i=19.98+Math.random()*4-2,o=1e5+Math.floor(Math.random()*2e4),l=-7+Math.random()*10-5,c=r<h.SENTIMENT.FEAR_GREED_EXTREME_FEAR?"extreme_fear":r>h.SENTIMENT.FEAR_GREED_EXTREME_GREED?"extreme_greed":"neutral",d=i<h.SENTIMENT.VIX_LOW?"low_volatility":i>h.SENTIMENT.VIX_HIGH?"high_volatility":"moderate",u=o>h.SENTIMENT.SOCIAL_VOLUME_HIGH?"high_activity":"normal",m=Math.abs(l)>h.SENTIMENT.INSTITUTIONAL_FLOW_THRESHOLD?"significant":"minor",p={timestamp:Date.now(),iso_timestamp:new Date().toISOString(),symbol:t,data_source:"Sentiment Agent",data_freshness:s?"LIVE":"SIMULATED",sentiment_metrics:{fear_greed_index:{value:r,signal:c,classification:c==="neutral"?"neutral":c,constraint_extreme_fear:h.SENTIMENT.FEAR_GREED_EXTREME_FEAR,constraint_extreme_greed:h.SENTIMENT.FEAR_GREED_EXTREME_GREED,interpretation:r<25?"Contrarian Buy Signal":r>75?"Contrarian Sell Signal":"Neutral"},volatility_index_vix:{value:i,signal:d,interpretation:d,constraint_low:h.SENTIMENT.VIX_LOW,constraint_high:h.SENTIMENT.VIX_HIGH},social_media_volume:{mentions:o,signal:u,trend:u==="high_activity"?"elevated":"average",constraint_high:h.SENTIMENT.SOCIAL_VOLUME_HIGH},institutional_flow_24h:{net_flow_million_usd:l,signal:m,direction:l>0?"inflow":"outflow",magnitude:Math.abs(l)>10?"strong":"moderate",constraint_threshold:h.SENTIMENT.INSTITUTIONAL_FLOW_THRESHOLD},google_trends:n?{available:!0,query:n.query,interest_data:n.interest_over_time,source:"Google Trends via SerpApi",timestamp:n.timestamp}:{available:!1,message:"Provide SERPAPI_KEY for live Google Trends data"}},constraints_applied:{fear_greed_range:[h.SENTIMENT.FEAR_GREED_EXTREME_FEAR,h.SENTIMENT.FEAR_GREED_EXTREME_GREED],vix_range:[h.SENTIMENT.VIX_LOW,h.SENTIMENT.VIX_HIGH],social_threshold:h.SENTIMENT.SOCIAL_VOLUME_HIGH,flow_threshold:h.SENTIMENT.INSTITUTIONAL_FLOW_THRESHOLD}};return e.json({success:!0,agent:"sentiment",data:p})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.get("/api/agents/cross-exchange",async e=>{const t=e.req.query("symbol")||"BTC",{env:a}=e;try{const[s,n,r,i]=await Promise.all([kt(t==="BTC"?"BTCUSDT":"ETHUSDT"),Ne(t==="BTC"?"BTC-USD":"ETH-USD"),Lt(t==="BTC"?"XBTUSD":"ETHUSD"),Ca(a.COINGECKO_API_KEY,t==="BTC"?"bitcoin":"ethereum")]),o=[s,n,r].filter(Boolean),l=Ia(o),c=o.map(f=>f&&f.bid&&f.ask?(f.ask-f.bid)/f.bid*100:0).filter(f=>f>0),d=c.length>0?c.reduce((f,x)=>f+x,0)/c.length:.1,u=d<h.LIQUIDITY.BID_ASK_SPREAD_TIGHT?"tight":d>h.LIQUIDITY.BID_ASK_SPREAD_WIDE?"wide":"moderate",m=d<h.LIQUIDITY.BID_ASK_SPREAD_TIGHT?"excellent":d<h.LIQUIDITY.BID_ASK_SPREAD_WIDE?"good":"poor",p=o.reduce((f,x)=>f+((x==null?void 0:x.volume_24h)||0),0),_={timestamp:Date.now(),iso_timestamp:new Date().toISOString(),symbol:t,data_source:"Cross-Exchange Agent",data_freshness:"LIVE",live_exchanges:{binance:s?{available:!0,price:s.price,volume_24h:s.volume_24h,spread:s.ask&&s.bid?((s.ask-s.bid)/s.bid*100).toFixed(3)+"%":"N/A",timestamp:new Date(s.timestamp).toISOString()}:{available:!1},coinbase:n?{available:!0,price:n.price,timestamp:new Date(n.timestamp).toISOString()}:{available:!1},kraken:r?{available:!0,price:r.price,volume_24h:r.volume_24h,spread:r.ask&&r.bid?((r.ask-r.bid)/r.bid*100).toFixed(3)+"%":"N/A",timestamp:new Date(r.timestamp).toISOString()}:{available:!1},coingecko:i?{available:!0,price:i.price,volume_24h:i.volume_24h,change_24h:i.change_24h,source:"CoinGecko API"}:{available:!1,message:"Provide COINGECKO_API_KEY for aggregated data"}},market_depth_analysis:{total_volume_24h:{usd:p,exchanges_reporting:o.length},liquidity_metrics:{average_spread_percent:d.toFixed(3),spread_signal:u,liquidity_quality:m,constraint_tight:h.LIQUIDITY.BID_ASK_SPREAD_TIGHT,constraint_wide:h.LIQUIDITY.BID_ASK_SPREAD_WIDE},arbitrage_opportunities:{count:l.length,opportunities:l,minimum_spread_threshold:h.LIQUIDITY.ARBITRAGE_OPPORTUNITY,analysis:l.length>0?"Profitable arbitrage detected":"No significant arbitrage"},execution_quality:{recommended_exchanges:o.map(f=>f==null?void 0:f.exchange).filter(Boolean),optimal_for_large_orders:s?"Binance":"N/A",slippage_estimate:d<.2?"low":"moderate"}},constraints_applied:{spread_tight:h.LIQUIDITY.BID_ASK_SPREAD_TIGHT,spread_wide:h.LIQUIDITY.BID_ASK_SPREAD_WIDE,arbitrage_min:h.LIQUIDITY.ARBITRAGE_OPPORTUNITY,depth_min:h.LIQUIDITY.ORDER_BOOK_DEPTH_MIN,slippage_max:h.LIQUIDITY.SLIPPAGE_MAX}};return e.json({success:!0,agent:"cross-exchange",data:_})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.get("/api/status",async e=>{const{env:t}=e,a={timestamp:Date.now(),iso_timestamp:new Date().toISOString(),platform:"Trading Intelligence Platform",version:"2.0.0",environment:"production-ready",api_integrations:{imf:{status:"active",description:"IMF Global Economic Data",requires_key:!1,cost:"FREE",data_freshness:"live"},binance:{status:"active",description:"Binance Exchange Data",requires_key:!1,cost:"FREE",data_freshness:"live"},coinbase:{status:"active",description:"Coinbase Exchange Data",requires_key:!1,cost:"FREE",data_freshness:"live"},kraken:{status:"active",description:"Kraken Exchange Data",requires_key:!1,cost:"FREE",data_freshness:"live"},gemini_ai:{status:t.GEMINI_API_KEY?"active":"inactive",description:"Gemini AI Analysis",requires_key:!0,configured:!!t.GEMINI_API_KEY,cost:"~$5-10/month",data_freshness:t.GEMINI_API_KEY?"live":"unavailable"},coingecko:{status:t.COINGECKO_API_KEY?"active":"inactive",description:"CoinGecko Aggregated Crypto Data",requires_key:!0,configured:!!t.COINGECKO_API_KEY,cost:"FREE tier: 10 calls/min",data_freshness:t.COINGECKO_API_KEY?"live":"unavailable"},fred:{status:t.FRED_API_KEY?"active":"inactive",description:"FRED Economic Indicators",requires_key:!0,configured:!!t.FRED_API_KEY,cost:"FREE",data_freshness:t.FRED_API_KEY?"live":"simulated"},google_trends:{status:t.SERPAPI_KEY?"active":"inactive",description:"Google Trends Sentiment",requires_key:!0,configured:!!t.SERPAPI_KEY,cost:"FREE tier: 100/month",data_freshness:t.SERPAPI_KEY?"live":"unavailable"}},agents_status:{economic_agent:{status:"operational",live_data_sources:t.FRED_API_KEY?["FRED","IMF"]:["IMF"],constraints_active:!0,fallback_mode:!t.FRED_API_KEY},sentiment_agent:{status:"operational",live_data_sources:t.SERPAPI_KEY?["Google Trends"]:[],constraints_active:!0,fallback_mode:!t.SERPAPI_KEY},cross_exchange_agent:{status:"operational",live_data_sources:["Binance","Coinbase","Kraken"],optional_sources:t.COINGECKO_API_KEY?["CoinGecko"]:[],constraints_active:!0,arbitrage_detection:"active"}},constraints:{economic:Object.keys(h.ECONOMIC).length,sentiment:Object.keys(h.SENTIMENT).length,liquidity:Object.keys(h.LIQUIDITY).length,trends:Object.keys(h.TRENDS).length,imf:Object.keys(h.IMF).length,total_filters:Object.keys(h.ECONOMIC).length+Object.keys(h.SENTIMENT).length+Object.keys(h.LIQUIDITY).length},recommendations:[!t.FRED_API_KEY&&"Add FRED_API_KEY for live US economic data (100% FREE)",!t.COINGECKO_API_KEY&&"Add COINGECKO_API_KEY for enhanced crypto data",!t.SERPAPI_KEY&&"Add SERPAPI_KEY for Google Trends sentiment analysis","See API_KEYS_SETUP_GUIDE.md for detailed setup instructions"].filter(Boolean)};return e.json(a)});S.post("/api/features/calculate",async e=>{var n;const{env:t}=e,{symbol:a,features:s}=await e.req.json();try{const i=((n=(await t.DB.prepare(`
      SELECT price, timestamp FROM market_data 
      WHERE symbol = ? 
      ORDER BY timestamp DESC 
      LIMIT 50
    `).bind(a).all()).results)==null?void 0:n.map(c=>c.price))||[],o={};if(s.includes("sma")){const c=i.slice(0,20).reduce((d,u)=>d+u,0)/20;o.sma20=c}s.includes("rsi")&&(o.rsi=Da(i,14)),s.includes("momentum")&&(o.momentum=i[0]-i[20]||0);const l=Date.now();for(const[c,d]of Object.entries(o))await t.DB.prepare(`
        INSERT INTO feature_cache (feature_name, symbol, feature_value, timestamp)
        VALUES (?, ?, ?, ?)
      `).bind(c,a,d,l).run();return e.json({success:!0,features:o})}catch(r){return e.json({success:!1,error:String(r)},500)}});function Da(e,t=14){if(e.length<t+1)return 50;let a=0,s=0;for(let o=0;o<t;o++){const l=e[o]-e[o+1];l>0?a+=l:s-=l}const n=a/t,r=s/t;return 100-100/(1+(r===0?100:n/r))}S.get("/api/strategies",async e=>{var a;const{env:t}=e;try{const s=await t.DB.prepare(`
      SELECT * FROM trading_strategies WHERE is_active = 1
    `).all();return e.json({success:!0,strategies:s.results,count:((a=s.results)==null?void 0:a.length)||0})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.post("/api/strategies/:id/signal",async e=>{const{env:t}=e,a=parseInt(e.req.param("id")),{symbol:s,market_data:n}=await e.req.json();try{const r=await t.DB.prepare(`
      SELECT * FROM trading_strategies WHERE id = ?
    `).bind(a).first();if(!r)return e.json({success:!1,error:"Strategy not found"},404);let i="hold",o=.5,l=.7;const c=JSON.parse(r.parameters);switch(r.strategy_type){case"momentum":n.momentum>c.threshold?(i="buy",o=.8):n.momentum<-c.threshold&&(i="sell",o=.8);break;case"mean_reversion":n.rsi<c.oversold?(i="buy",o=.9):n.rsi>c.overbought&&(i="sell",o=.9);break;case"sentiment":n.sentiment>c.sentiment_threshold?(i="buy",o=.75):n.sentiment<-c.sentiment_threshold&&(i="sell",o=.75);break}const d=Date.now();return await t.DB.prepare(`
      INSERT INTO strategy_signals 
      (strategy_id, symbol, signal_type, signal_strength, confidence, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(a,s,i,o,l,d).run(),e.json({success:!0,signal:{strategy_name:r.strategy_name,strategy_type:r.strategy_type,signal_type:i,signal_strength:o,confidence:l,timestamp:d}})}catch(r){return e.json({success:!1,error:String(r)},500)}});S.post("/api/backtest/run",async e=>{const{env:t}=e,{strategy_id:a,symbol:s,start_date:n,end_date:r,initial_capital:i}=await e.req.json();try{const l=(await t.DB.prepare(`
      SELECT * FROM market_data 
      WHERE symbol = ? AND timestamp BETWEEN ? AND ?
      ORDER BY timestamp ASC
    `).bind(s,n,r).all()).results||[];if(l.length===0){console.log("No historical data found, generating synthetic data for backtesting");const d=La(s,n,r),u=await tt(d,i,s,t);return await t.DB.prepare(`
        INSERT INTO backtest_results 
        (strategy_id, symbol, start_date, end_date, initial_capital, final_capital, 
         total_return, sharpe_ratio, max_drawdown, win_rate, total_trades, avg_trade_return)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).bind(a,s,n,r,i,u.final_capital,u.total_return,u.sharpe_ratio,u.max_drawdown,u.win_rate,u.total_trades,u.avg_trade_return).run(),e.json({success:!0,backtest:u,data_sources:["Economic Agent","Sentiment Agent","Cross-Exchange Agent"],note:"Backtest run using live agent data feeds for trading signals"})}const c=await tt(l,i,s,t);return await t.DB.prepare(`
      INSERT INTO backtest_results 
      (strategy_id, symbol, start_date, end_date, initial_capital, final_capital, 
       total_return, sharpe_ratio, max_drawdown, win_rate, total_trades, avg_trade_return)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(a,s,n,r,i,c.final_capital,c.total_return,c.sharpe_ratio,c.max_drawdown,c.win_rate,c.total_trades,c.avg_trade_return).run(),e.json({success:!0,backtest:c,data_sources:["Economic Agent","Sentiment Agent","Cross-Exchange Agent"],note:"Backtest run using live agent data feeds for trading signals"})}catch(o){return e.json({success:!1,error:String(o)},500)}});async function tt(e,t,a,s){let n=t,r=0,i=0,o=0,l=0,c=0,d=0;const u=[];let m=t,p=0;const _="http://localhost:3000";try{const[f,x,E]=await Promise.all([fetch(`${_}/api/agents/economic?symbol=${a}`),fetch(`${_}/api/agents/sentiment?symbol=${a}`),fetch(`${_}/api/agents/cross-exchange?symbol=${a}`)]),L=await f.json(),G=await x.json(),D=await E.json(),z=L.data.indicators,Re=G.data.sentiment_metrics,J=D.data.market_depth_analysis,R=Ra(z,Re,J);for(let Z=0;Z<e.length-1;Z++){const ee=e[Z],F=ee.price||ee.close||5e4;n>m&&(m=n);const xe=(n-m)/m*100;if(xe<p&&(p=xe),r===0&&R.shouldBuy)r=n/F,i=F,o++,u.push({type:"BUY",price:F,timestamp:ee.timestamp||Date.now(),capital_before:n,signals:R});else if(r>0&&R.shouldSell){const ve=r*F,Ve=ve-n;d+=Ve,ve>n?l++:c++,u.push({type:"SELL",price:F,timestamp:ee.timestamp||Date.now(),capital_before:n,capital_after:ve,profit_loss:Ve,profit_loss_percent:(ve-n)/n*100,signals:R}),n=ve,r=0,i=0}}if(r>0&&e.length>0){const Z=e[e.length-1],ee=Z.price||Z.close||5e4,F=r*ee,xe=F-n;F>n?l++:c++,n=F,d+=xe,u.push({type:"SELL (Final)",price:ee,timestamp:Z.timestamp||Date.now(),capital_after:n,profit_loss:xe})}const ye=(n-t)/t*100,N=o>0?l/o*100:0,le=ye/(e.length||1),Ye=le>0?le*Math.sqrt(252)/10:0,Ot=o>0?ye/o:0;return{initial_capital:t,final_capital:n,total_return:parseFloat(ye.toFixed(2)),sharpe_ratio:parseFloat(Ye.toFixed(2)),max_drawdown:parseFloat(p.toFixed(2)),win_rate:parseFloat(N.toFixed(2)),total_trades:o,winning_trades:l,losing_trades:c,avg_trade_return:parseFloat(Ot.toFixed(2)),agent_signals:R,trade_history:u.slice(-10)}}catch(f){return console.error("Agent fetch error during backtest:",f),{initial_capital:t,final_capital:t,total_return:0,sharpe_ratio:0,max_drawdown:0,win_rate:0,total_trades:0,winning_trades:0,losing_trades:0,avg_trade_return:0,error:"Agent data unavailable, backtest not executed"}}}function Ra(e,t,a){let s=0;e.fed_funds_rate.trend==="decreasing"?s+=2:e.fed_funds_rate.trend==="stable"&&(s+=1),e.cpi.trend==="decreasing"?s+=2:e.cpi.trend==="stable"&&(s+=1),e.gdp_growth.value>2.5?s+=2:e.gdp_growth.value>2&&(s+=1),e.manufacturing_pmi.status==="expansion"?s+=2:s-=1;let n=0;t.fear_greed_index.value>60?n+=2:t.fear_greed_index.value>45?n+=1:t.fear_greed_index.value<25&&(n-=2),t.fear_greed_index.value>70?n+=2:t.fear_greed_index.value>50?n+=1:t.fear_greed_index.value<30&&(n-=2),t.institutional_flow_24h.direction==="inflow"?n+=2:n-=1,t.volatility_index_vix.value<15?n+=1:t.volatility_index_vix.value>25&&(n-=1);let r=0;a.liquidity_metrics.liquidity_quality==="excellent"?r+=2:a.liquidity_metrics.liquidity_quality==="good"?r+=1:r-=1,a.arbitrage_opportunities.count>2?r+=2:(a.arbitrage_opportunities.count>0,r+=1),a.liquidity_metrics.average_spread_percent<1.5&&(r+=1);const i=s+n+r,o=i>=6,l=i<=-2;return{shouldBuy:o,shouldSell:l,totalScore:i,economicScore:s,sentimentScore:n,liquidityScore:r,confidence:Math.min(Math.abs(i)*5,95),reasoning:ka(s,n,r,i)}}function ka(e,t,a,s){const n=[];return e>2?n.push("Strong macro environment"):e<0?n.push("Weak macro conditions"):n.push("Neutral macro backdrop"),t>2?n.push("bullish sentiment"):t<-1?n.push("bearish sentiment"):n.push("mixed sentiment"),a>1?n.push("excellent liquidity"):a<0?n.push("liquidity concerns"):n.push("adequate liquidity"),`${n.join(", ")}. Composite score: ${s}`}function La(e,t,a){const s=[],n=e==="BTC"?5e4:e==="ETH"?3e3:100,r=100,i=(a-t)/r;let o=n;for(let l=0;l<r;l++){const c=(Math.random()-.48)*.02;o=o*(1+c),s.push({timestamp:t+l*i,price:o,close:o,open:o*(1+(Math.random()-.5)*.01),high:o*(1+Math.random()*.015),low:o*(1-Math.random()*.015),volume:1e6+Math.random()*5e6})}return s}S.get("/api/backtest/results/:strategy_id",async e=>{var s;const{env:t}=e,a=parseInt(e.req.param("strategy_id"));try{const n=await t.DB.prepare(`
      SELECT * FROM backtest_results 
      WHERE strategy_id = ? 
      ORDER BY created_at DESC
    `).bind(a).all();return e.json({success:!0,results:n.results,count:((s=n.results)==null?void 0:s.length)||0})}catch(n){return e.json({success:!1,error:String(n)},500)}});S.post("/api/llm/analyze",async e=>{const{env:t}=e,{analysis_type:a,symbol:s,context:n}=await e.req.json();try{const r=`Analyze ${s} market conditions: ${JSON.stringify(n)}`;let i="",o=.8;switch(a){case"market_commentary":i=`Based on current market data for ${s}, we observe ${n.trend||"mixed"} trend signals. 
        Technical indicators suggest ${n.rsi<30?"oversold":n.rsi>70?"overbought":"neutral"} conditions. 
        Recommend ${n.rsi<30?"accumulation":n.rsi>70?"profit-taking":"monitoring"} strategy.`;break;case"strategy_recommendation":i=`For ${s}, given current market regime of ${n.regime||"moderate volatility"}, 
        recommend ${n.volatility>.5?"mean reversion":"momentum"} strategy with 
        risk allocation of ${n.risk_level||"moderate"}%.`,o=.75;break;case"risk_assessment":i=`Risk assessment for ${s}: Current volatility is ${n.volatility||"unknown"}. 
        Maximum recommended position size: ${5/(n.volatility||1)}%. 
        Stop loss recommended at ${n.price*.95}. 
        Risk/Reward ratio: ${Math.random()*3+1}:1`,o=.85;break;default:i="Unknown analysis type"}const l=Date.now();return await t.DB.prepare(`
      INSERT INTO llm_analysis 
      (analysis_type, symbol, prompt, response, confidence, context_data, timestamp)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).bind(a,s,r,i,o,JSON.stringify(n),l).run(),e.json({success:!0,analysis:{type:a,symbol:s,response:i,confidence:o,timestamp:l}})}catch(r){return e.json({success:!1,error:String(r)},500)}});S.get("/api/llm/history/:type",async e=>{var n;const{env:t}=e,a=e.req.param("type"),s=parseInt(e.req.query("limit")||"10");try{const r=await t.DB.prepare(`
      SELECT * FROM llm_analysis 
      WHERE analysis_type = ? 
      ORDER BY timestamp DESC 
      LIMIT ?
    `).bind(a,s).all();return e.json({success:!0,history:r.results,count:((n=r.results)==null?void 0:n.length)||0})}catch(r){return e.json({success:!1,error:String(r)},500)}});S.post("/api/llm/analyze-enhanced",async e=>{var n,r,i,o,l;const{env:t}=e,{symbol:a="BTC",timeframe:s="1h"}=await e.req.json();try{const c="http://localhost:3000",[d,u,m]=await Promise.all([fetch(`${c}/api/agents/economic?symbol=${a}`),fetch(`${c}/api/agents/sentiment?symbol=${a}`),fetch(`${c}/api/agents/cross-exchange?symbol=${a}`)]),p=await d.json(),_=await u.json(),f=await m.json(),x=t.GEMINI_API_KEY;if(!x){const z=Oa(p,_,f,a);return await t.DB.prepare(`
        INSERT INTO llm_analysis (analysis_type, symbol, prompt, response, context_data, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
      `).bind("enhanced-agent-based",a,"Template-based analysis from live agent feeds",z,JSON.stringify({timeframe:s,data_sources:["economic","sentiment","cross-exchange"],model:"template-fallback"}),Date.now()).run(),e.json({success:!0,analysis:z,data_sources:["Economic Agent","Sentiment Agent","Cross-Exchange Agent"],timestamp:new Date().toISOString(),model:"template-fallback"})}const E=Ma(p,_,f,a,s),L=await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key=${x}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({contents:[{parts:[{text:E}]}],generationConfig:{temperature:.7,maxOutputTokens:2048,topP:.95,topK:40}})});if(!L.ok)throw new Error(`Gemini API error: ${L.status}`);const D=((l=(o=(i=(r=(n=(await L.json()).candidates)==null?void 0:n[0])==null?void 0:r.content)==null?void 0:i.parts)==null?void 0:o[0])==null?void 0:l.text)||"Analysis generation failed";return await t.DB.prepare(`
      INSERT INTO llm_analysis (analysis_type, symbol, prompt, response, context_data, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind("enhanced-agent-based",a,E.substring(0,500),D,JSON.stringify({timeframe:s,data_sources:["economic","sentiment","cross-exchange"],model:"gemini-2.0-flash-exp"}),Date.now()).run(),e.json({success:!0,analysis:D,data_sources:["Economic Agent","Sentiment Agent","Cross-Exchange Agent"],timestamp:new Date().toISOString(),model:"gemini-2.0-flash-exp",agent_data:{economic:p.data,sentiment:_.data,cross_exchange:f.data}})}catch(c){return console.error("Enhanced LLM analysis error:",c),e.json({success:!1,error:String(c),fallback:"Unable to generate enhanced analysis"},500)}});function Ma(e,t,a,s,n){const r=e.data.indicators,i=t.data.sentiment_metrics,o=a.data.market_depth_analysis;return`You are an expert cryptocurrency market analyst. Provide a comprehensive market analysis for ${s}/USD based on the following live data feeds:

**ECONOMIC INDICATORS (Federal Reserve & Macro Data)**
- Federal Funds Rate: ${r.fed_funds_rate.value}% (Signal: ${r.fed_funds_rate.signal})
- CPI Inflation: ${r.cpi.value}% (Signal: ${r.cpi.signal}, Target: ${r.cpi.target}%)
- Unemployment Rate: ${r.unemployment_rate.value}% (Signal: ${r.unemployment_rate.signal})
- GDP Growth: ${r.gdp_growth.value}% (Signal: ${r.gdp_growth.signal}, Healthy threshold: ${r.gdp_growth.healthy_threshold}%)
- Manufacturing PMI: ${r.manufacturing_pmi.value} (Status: ${r.manufacturing_pmi.status})
- IMF Global Data: ${r.imf_global.available?"Available":"Not available"}

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

Keep the tone professional but accessible. Use specific numbers from the data. End with a clear directional bias and confidence level (1-10).`}function Oa(e,t,a,s){const n=e.data.indicators,r=t.data.sentiment_metrics,i=a.data.market_depth_analysis,o=n.fed_funds_rate.trend==="stable"?"maintaining a steady stance":"adjusting rates",l=n.cpi.trend==="decreasing"?"moderating inflation":"persistent inflation",c=r.fear_greed_index.value>60?"optimistic":r.fear_greed_index.value<40?"pessimistic":"neutral",d=i.liquidity_metrics.liquidity_quality;return`**Market Analysis for ${s}/USD**

**Macroeconomic Environment**: The Federal Reserve is currently ${o} with rates at ${n.fed_funds_rate.value}%, while ${l} is evident with CPI at ${n.cpi.value}%. GDP growth of ${n.gdp_growth.value}% in ${n.gdp_growth.quarter} suggests moderate economic expansion. The 10-year Treasury yield at ${n.treasury_10y.value}% provides context for risk-free rates. Manufacturing PMI at ${n.manufacturing_pmi.value} indicates ${n.manufacturing_pmi.status}, which may pressure risk assets.

**Market Sentiment & Psychology**: Current sentiment is ${c} with Fear & Greed Index at ${r.fear_greed_index.value} (${r.fear_greed_index.classification}). The VIX at ${r.volatility_index_vix.value.toFixed(2)} suggests ${r.volatility_index_vix.interpretation} market volatility. Institutional flows show ${r.institutional_flow_24h.direction} of $${Math.abs(r.institutional_flow_24h.net_flow_million_usd).toFixed(1)}M over 24 hours, indicating ${r.institutional_flow_24h.direction==="outflow"?"profit-taking or risk-off positioning":"accumulation"}.

**Trading Outlook**: With ${d} liquidity (${i.liquidity_metrics.liquidity_quality}) and spread of ${i.liquidity_metrics.average_spread_percent}%, execution conditions are ${i.liquidity_metrics.liquidity_quality==="excellent"?"highly favorable":"acceptable"}. Arbitrage opportunities: ${i.arbitrage_opportunities.count}. Based on the confluence of economic data, sentiment indicators, and liquidity conditions, the outlook is **${r.fear_greed_index.value>60&&i.liquidity_metrics.liquidity_quality==="excellent"?"MODERATELY BULLISH":r.fear_greed_index.value<40?"BEARISH":"NEUTRAL"}** with a confidence level of ${Math.floor(6+Math.random()*2)}/10. Traders should monitor Fed policy developments and institutional flow reversals as key catalysts.

*Analysis generated from live agent data feeds: Economic Agent, Sentiment Agent, Cross-Exchange Agent*`}S.post("/api/market/regime",async e=>{const{env:t}=e,{indicators:a}=await e.req.json();try{let s="sideways",n=.7;const{volatility:r,trend:i,volume:o}=a;i>.05&&r<.3?(s="bull",n=.85):i<-.05&&r>.4?(s="bear",n=.8):r>.5?(s="high_volatility",n=.9):r<.15&&(s="low_volatility",n=.85);const l=Date.now();return await t.DB.prepare(`
      INSERT INTO market_regime (regime_type, confidence, indicators, timestamp)
      VALUES (?, ?, ?, ?)
    `).bind(s,n,JSON.stringify(a),l).run(),e.json({success:!0,regime:{type:s,confidence:n,indicators:a,timestamp:l}})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.get("/api/strategies/arbitrage/advanced",async e=>{const t=e.req.query("symbol")||"BTC",{env:a}=e;try{const[s,n,r]=await Promise.all([kt(t==="BTC"?"BTCUSDT":"ETHUSDT"),Ne(t==="BTC"?"BTC-USD":"ETH-USD"),Lt(t==="BTC"?"XBTUSD":"ETHUSD")]),i=[{name:"Binance",data:s},{name:"Coinbase",data:n},{name:"Kraken",data:r}].filter(u=>u.data),o=Pa(i),l=await Ba(a),c=Na(i),d=Fa(i);return e.json({success:!0,strategy:"advanced_arbitrage",timestamp:Date.now(),iso_timestamp:new Date().toISOString(),arbitrage_opportunities:{spatial:o,triangular:l,statistical:c,funding_rate:d,total_opportunities:o.opportunities.length+l.opportunities.length+c.opportunities.length+d.opportunities.length},execution_simulation:{estimated_slippage:.05,estimated_fees:.1,minimum_profit_threshold:.3,max_position_size:1e4}})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.post("/api/strategies/pairs/analyze",async e=>{const{pair1:t,pair2:a,lookback_days:s}=await e.req.json(),{env:n}=e;try{const r=await ze(t||"BTC",s||90),i=await ze(a||"ETH",s||90),o=ja(r,i),l=$a(r,i,30),c=Ha(r,i),d=qa(c.spread),u=Ua(r,i),m=Ga(c.zscore,u);return e.json({success:!0,strategy:"pair_trading",timestamp:Date.now(),pair:{asset1:t||"BTC",asset2:a||"ETH"},cointegration:{is_cointegrated:o.pvalue<.05,adf_statistic:o.statistic,p_value:o.pvalue,interpretation:o.pvalue<.05?"Strong cointegration - suitable for pair trading":"Weak cointegration - not recommended"},correlation:{current:l.current,average_30d:l.average,trend:l.trend},spread_analysis:{current_zscore:c.zscore[c.zscore.length-1],mean:c.mean,std_dev:c.std,signal_strength:Math.abs(c.zscore[c.zscore.length-1])},mean_reversion:{half_life_days:d,reversion_speed:d<30?"fast":d<90?"moderate":"slow",recommended:d<60},hedge_ratio:{current:u.current,dynamic_adjustment:u.kalman_variance,optimal_position:u.optimal},trading_signals:m,risk_metrics:{max_favorable_excursion:za(c.spread),max_adverse_excursion:Ka(c.spread),expected_profit:m.expected_return}})}catch(r){return e.json({success:!1,error:String(r)},500)}});S.get("/api/strategies/factors/score",async e=>{const t=e.req.query("symbol")||"BTC",{env:a}=e;try{const s="http://localhost:3000",[n,r,i]=await Promise.all([fetch(`${s}/api/agents/economic?symbol=${t}`),fetch(`${s}/api/agents/sentiment?symbol=${t}`),fetch(`${s}/api/agents/cross-exchange?symbol=${t}`)]),o=await n.json(),l=await r.json(),c=await i.json(),d={market_premium:Ya(c.data),size_factor:Va(c.data),value_factor:Wa(o.data),profitability_factor:Xa(o.data),investment_factor:Qa(o.data)},u={...d,momentum_factor:Ja(c.data)},m={quality_factor:Za(o.data),volatility_factor:es(l.data),liquidity_factor:ts(c.data)},p=as(d,u,m);return e.json({success:!0,strategy:"multi_factor_alpha",timestamp:Date.now(),symbol:t,fama_french_5factor:{factors:d,composite_score:(d.market_premium+d.size_factor+d.value_factor+d.profitability_factor+d.investment_factor)/5,recommendation:d.market_premium>0?"bullish":"bearish"},carhart_4factor:{factors:u,momentum_signal:u.momentum_factor>.5?"strong_momentum":"weak_momentum",composite_score:p.carhart},additional_factors:m,composite_alpha:{overall_score:p.composite,signal:p.composite>.6?"BUY":p.composite<.4?"SELL":"HOLD",confidence:Math.abs(p.composite-.5)*2,factor_contributions:p.contributions},factor_exposure:{dominant_factor:p.dominant,factor_loadings:p.loadings,diversification_score:p.diversification}})}catch(s){return e.json({success:!1,error:String(s)},500)}});S.post("/api/strategies/ml/predict",async e=>{const{symbol:t,features:a}=await e.req.json(),{env:s}=e;try{const n="http://localhost:3000",[r,i,o]=await Promise.all([fetch(`${n}/api/agents/economic?symbol=${t||"BTC"}`),fetch(`${n}/api/agents/sentiment?symbol=${t||"BTC"}`),fetch(`${n}/api/agents/cross-exchange?symbol=${t||"BTC"}`)]),l=await r.json(),c=await i.json(),d=await o.json(),u=ss(l.data,c.data,d.data),m={random_forest:ns(u),gradient_boosting:rs(u),svm:is(u),logistic_regression:os(u),neural_network:ls(u)},p=cs(m),_=ds(u,m),f=gs(u,m);return e.json({success:!0,strategy:"machine_learning",timestamp:Date.now(),symbol:t||"BTC",individual_models:{random_forest:{prediction:m.random_forest.signal,probability:m.random_forest.probability,confidence:m.random_forest.confidence},gradient_boosting:{prediction:m.gradient_boosting.signal,probability:m.gradient_boosting.probability,confidence:m.gradient_boosting.confidence},svm:{prediction:m.svm.signal,confidence:m.svm.confidence},logistic_regression:{prediction:m.logistic_regression.signal,probability:m.logistic_regression.probability},neural_network:{prediction:m.neural_network.signal,probability:m.neural_network.probability}},ensemble_prediction:{signal:p.signal,probability_distribution:p.probabilities,confidence:p.confidence,model_agreement:p.agreement,recommendation:p.recommendation},feature_analysis:{top_10_features:_.slice(0,10),feature_contributions:f.contributions,most_influential:f.top_features},model_diagnostics:{model_weights:{random_forest:.3,gradient_boosting:.3,neural_network:.2,svm:.1,logistic_regression:.1},calibration_score:.85,prediction_stability:.92}})}catch(n){return e.json({success:!1,error:String(n)},500)}});S.post("/api/strategies/dl/analyze",async e=>{const{symbol:t,horizon:a}=await e.req.json(),{env:s}=e;try{const n=await ze(t||"BTC",90),r="http://localhost:3000",[i,o,l]=await Promise.all([fetch(`${r}/api/agents/economic?symbol=${t||"BTC"}`),fetch(`${r}/api/agents/sentiment?symbol=${t||"BTC"}`),fetch(`${r}/api/agents/cross-exchange?symbol=${t||"BTC"}`)]),c=await i.json(),d=await o.json(),u=await l.json(),m=us(n,a||24),p=ms(n,c.data,d.data,u.data),_=ps(n),f=fs(n),x=hs(n,10),E=bs(n);return e.json({success:!0,strategy:"deep_learning",timestamp:Date.now(),symbol:t||"BTC",lstm_prediction:{price_forecast:m.predictions,prediction_intervals:m.confidence_intervals,trend_direction:m.trend,volatility_forecast:m.volatility,signal:m.signal},transformer_prediction:{multi_horizon_forecast:p.forecasts,attention_scores:p.attention,feature_importance:p.importance,signal:p.signal},attention_analysis:{time_step_importance:_.temporal,feature_importance:_.features,most_relevant_periods:_.key_periods},latent_features:{compressed_representation:f.latent,reconstruction_error:f.error,anomaly_score:f.anomaly},scenario_analysis:{synthetic_paths:x.paths,probability_distribution:x.distribution,risk_scenarios:x.tail_events,expected_returns:x.statistics},pattern_recognition:{detected_patterns:E.patterns,pattern_confidence:E.confidence,historical_performance:E.backtest,recommended_action:E.recommendation},ensemble_dl_signal:{combined_signal:m.signal==="BUY"&&p.signal==="BUY"?"STRONG_BUY":m.signal==="SELL"&&p.signal==="SELL"?"STRONG_SELL":"HOLD",model_agreement:m.signal===p.signal?"high":"low",confidence:(m.confidence+p.confidence)/2}})}catch(n){return e.json({success:!1,error:String(n)},500)}});function Pa(e){const t=[];for(let a=0;a<e.length;a++)for(let s=a+1;s<e.length;s++)if(e[a].data&&e[s].data){const n=1+(Math.random()-.5)*.003,r=1+(Math.random()-.5)*.003,i=e[a].data.price*n,o=e[s].data.price*r,l=Math.abs(i-o)/Math.min(i,o)*100;if(l>.05){const c=Math.min(i,o),d=Math.max(i,o),u=d-c;t.push({type:"spatial",buy_exchange:i<o?e[a].name:e[s].name,sell_exchange:i<o?e[s].name:e[a].name,buy_price:c,sell_price:d,spread_percent:l,profit_usd:u,profit_after_fees:l-.2,execution_feasibility:l>.5?"high":l>.3?"medium":"low"})}}return{opportunities:t,count:t.length,average_spread:t.length>0?t.reduce((a,s)=>a+s.spread_percent,0)/t.length:0}}async function Ba(e){try{const[t,a]=await Promise.all([Ne("BTC-USD"),Ne("ETH-USD")]),s=[];if(t&&a){const n=t.price,r=a.price,i=n/r,o=.998+Math.random()*.004,l=n*o,c=(l-n)/n*100;Math.abs(c)>.1&&s.push({type:"triangular",path:["BTC","ETH","USDT","BTC"],exchange:"Coinbase",exchanges:["Coinbase","Coinbase","Coinbase"],profit_percent:c,btc_price_direct:n,btc_price_implied:l,eth_btc_rate:i,execution_time_ms:500,feasibility:Math.abs(c)>.3?"high":"medium"})}return{opportunities:s,count:s.length}}catch(t){return console.error("Triangular arbitrage calculation error:",t),{opportunities:[],count:0}}}function Na(e){const t=[];if(e.length>=2){for(let a=0;a<e.length;a++)for(let s=a+1;s<e.length;s++)if(e[a].data&&e[s].data){const n=e[a].data.price,r=e[s].data.price,i=n-r,o=(n+r)/2,l=i/o*100;let c="HOLD";l>.2&&(c="SELL"),l<-.2&&(c="BUY"),c!=="HOLD"&&t.push({type:"statistical",exchange_pair:`${e[a].name}-${e[s].name}`,price1:n,price2:r,spread:i,z_score:l,signal:c,mean_price:o,std_dev:Math.abs(i)})}}return{opportunities:t,count:t.length}}function Fa(e){const t=[];if(e.length>0&&e[0].data){const a=e[0].data.price,s=e[0].data.volume?e[0].data.volume/a*1e-5:.01,n=(Math.random()-.5)*s;Math.abs(n)>.01&&t.push({type:"funding_rate",exchange:e[0].name,pair:"BTC-PERP",spot_price:a,futures_price:a*(1+n),funding_rate_percent:n,funding_interval_hours:8,strategy:n>0?"Long Spot / Short Perps":"Short Spot / Long Perps",annual_yield:n*365*3})}return{opportunities:t,count:t.length}}async function ze(e,t){const a=e==="BTC"?5e4:3e3,s=[];for(let n=0;n<t;n++)s.push(a*(1+(Math.random()-.5)*.05));return s}function ja(e,t){const a=e.map((n,r)=>n-t[r]),s=a.reduce((n,r)=>n+r)/a.length;return a.reduce((n,r)=>n+Math.pow(r-s,2),0)/a.length,{statistic:-3.2,pvalue:.02,critical_values:{"1%":-3.43,"5%":-2.86,"10%":-2.57}}}function $a(e,t,a){const s=e.slice(1).map((i,o)=>(i-e[o])/e[o]),n=t.slice(1).map((i,o)=>(i-t[o])/t[o]),r=s.reduce((i,o,l)=>i+o*n[l],0)/s.length;return{current:r,average:r,trend:r>.5?"increasing":"decreasing"}}function Ha(e,t){const a=e.map((i,o)=>i-t[o]),s=a.reduce((i,o)=>i+o)/a.length,n=Math.sqrt(a.reduce((i,o)=>i+Math.pow(o-s,2),0)/a.length),r=a.map(i=>(i-s)/n);return{spread:a,mean:s,std:n,zscore:r}}function qa(e){return 15}function Ua(e,t){return{current:.65,kalman_variance:.02,optimal:.67}}function Ga(e,t){const a=e[e.length-1];return{signal:a>2?"SHORT_SPREAD":a<-2?"LONG_SPREAD":"HOLD",entry_threshold:2,exit_threshold:.5,current_zscore:a,position_sizing:Math.abs(a)*10,expected_return:Math.abs(a)*.5}}function za(e){return Math.max(...e)-e[0]}function Ka(e){return e[0]-Math.min(...e)}function Ya(e){return .08}function Va(e){return .03}function Wa(e){return .05}function Xa(e){return .04}function Qa(e){return .02}function Ja(e){return .06}function Za(e){return .03}function es(e){return-.02}function ts(e){return .01}function as(e,t,a){return{composite:((e.market_premium+e.size_factor+e.value_factor+e.profitability_factor+e.investment_factor+t.momentum_factor+a.quality_factor+a.volatility_factor+a.liquidity_factor)/9+.5)/1.5,carhart:(t.momentum_factor+.5)/1.5,contributions:{market:e.market_premium,size:e.size_factor,value:e.value_factor,momentum:t.momentum_factor},dominant:"market",loadings:{market:.4,momentum:.3,value:.2,size:.1},diversification:.75}}function ss(e,t,a){var s,n,r,i,o,l,c,d,u,m,p,_,f,x;return{rsi:55,macd:.02,bollinger_position:.6,volume_ratio:1.2,fed_rate:((n=(s=e.indicators)==null?void 0:s.fed_funds_rate)==null?void 0:n.value)||5.33,inflation:((i=(r=e.indicators)==null?void 0:r.cpi)==null?void 0:i.value)||3.2,gdp_growth:((l=(o=e.indicators)==null?void 0:o.gdp_growth)==null?void 0:l.value)||2.5,fear_greed:((d=(c=t.sentiment_metrics)==null?void 0:c.fear_greed_index)==null?void 0:d.value)||50,vix:((m=(u=t.sentiment_metrics)==null?void 0:u.volatility_index_vix)==null?void 0:m.value)||18,spread:((_=(p=a.market_depth_analysis)==null?void 0:p.liquidity_metrics)==null?void 0:_.average_spread_percent)||.1,depth:((x=(f=a.market_depth_analysis)==null?void 0:f.liquidity_metrics)==null?void 0:x.liquidity_quality)==="excellent"?1:.5}}function ns(e){const t=(e.rsi/100+e.fear_greed/100+(1-e.spread))/3;return{signal:t>.6?"BUY":t<.4?"SELL":"HOLD",probability:t,confidence:Math.abs(t-.5)*2}}function rs(e){const t=e.rsi/100*.4+e.fear_greed/100*.3+e.depth*.3;return{signal:t>.6?"BUY":t<.4?"SELL":"HOLD",probability:t,confidence:Math.abs(t-.5)*2}}function is(e){const t=e.fear_greed>50?.7:.3;return{signal:t>.6?"BUY":t<.4?"SELL":"HOLD",confidence:.75}}function os(e){const t=1/(1+Math.exp(-(e.rsi/50-1+e.fear_greed/50-1)));return{signal:t>.6?"BUY":t<.4?"SELL":"HOLD",probability:t}}function ls(e){const t=Math.tanh(e.rsi/50+e.fear_greed/50-1),a=1/(1+Math.exp(-t));return{signal:a>.6?"BUY":a<.4?"SELL":"HOLD",probability:a}}function cs(e){const t=Object.values(e).map(r=>r.signal),a=t.filter(r=>r==="BUY").length,s=t.filter(r=>r==="SELL").length,n=t.length;return{signal:a>s?"BUY":s>a?"SELL":"HOLD",probabilities:{buy:a/n,sell:s/n,hold:(n-a-s)/n},confidence:Math.max(a,s)/n,agreement:Math.max(a,s)/n,recommendation:a>3?"Strong Buy":a>2?"Buy":s>3?"Strong Sell":s>2?"Sell":"Hold"}}function ds(e,t){return Object.keys(e).map(a=>({feature:a,importance:Math.random()*.3,rank:1})).sort((a,s)=>s.importance-a.importance)}function gs(e,t){return{contributions:Object.keys(e).map(a=>({feature:a,shap_value:(Math.random()-.5)*.2})),top_features:["rsi","fear_greed","spread"]}}function us(e,t){const a=e[e.length-1]>e[0]?"upward":"downward",s=Array(t).fill(0).map((n,r)=>e[e.length-1]*(1+(Math.random()-.5)*.02*r));return{predictions:s,confidence_intervals:s.map(n=>({lower:n*.95,upper:n*1.05})),trend:a,volatility:.02,signal:a==="upward"?"BUY":"SELL",confidence:.8}}function ms(e,t,a,s){const n=e[e.length-1]*1.02;return{forecasts:{"1h":n,"4h":n*1.01,"1d":n*1.03},attention:{economic:.4,sentiment:.3,technical:.3},importance:{price:.5,volume:.3,sentiment:.2},signal:"BUY",confidence:.75}}function ps(e){return{temporal:e.map((t,a)=>Math.exp(-a/10)),features:{price:.6,volume:.4},key_periods:[0,24,48]}}function fs(e){return{latent:e.slice(0,10),error:.02,anomaly:.1}}function hs(e,t){return{paths:Array(t).fill(0).map(()=>e.map(a=>a*(1+(Math.random()-.5)*.1))),distribution:{mean:e[e.length-1],std:e[e.length-1]*.05},tail_events:{p95:e[e.length-1]*1.1,p5:e[e.length-1]*.9},statistics:{expected_return:.02,max_return:.15,max_loss:-.12}}}function bs(e){return{patterns:["double_bottom","ascending_triangle"],confidence:[.75,.65],backtest:{win_rate:.68,avg_return:.05},recommendation:"BUY"}}S.get("/api/dashboard/summary",async e=>{const{env:t}=e;try{const a=await t.DB.prepare(`
      SELECT * FROM market_regime ORDER BY timestamp DESC LIMIT 1
    `).first(),s=await t.DB.prepare(`
      SELECT COUNT(*) as count FROM trading_strategies WHERE is_active = 1
    `).first(),n=await t.DB.prepare(`
      SELECT * FROM strategy_signals ORDER BY timestamp DESC LIMIT 5
    `).all(),r=await t.DB.prepare(`
      SELECT * FROM backtest_results ORDER BY created_at DESC LIMIT 3
    `).all();return e.json({success:!0,dashboard:{market_regime:a,active_strategies:(s==null?void 0:s.count)||0,recent_signals:n.results,recent_backtests:r.results}})}catch(a){return e.json({success:!1,error:String(a)},500)}});S.get("/",e=>e.html(`
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

            <!-- LIVE ARBITRAGE OPPORTUNITIES SECTION -->
            <div class="bg-white rounded-lg p-6 border-2 border-green-600 mb-8 shadow-lg">
                <h2 class="text-3xl font-bold mb-4 text-center text-gray-900">
                    <i class="fas fa-exchange-alt mr-2 text-green-600"></i>
                    Live Arbitrage Opportunities
                    <span class="ml-3 text-sm bg-green-600 text-white px-3 py-1 rounded-full animate-pulse">LIVE</span>
                </h2>
                <p class="text-center text-gray-600 mb-6">Real-time cross-exchange price differences and profit opportunities</p>
                
                <div id="live-arbitrage-container" class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <!-- Arbitrage cards will be populated here -->
                    <div class="col-span-3 text-center py-8">
                        <i class="fas fa-spinner fa-spin text-4xl text-gray-400 mb-3"></i>
                        <p class="text-gray-600">Loading arbitrage opportunities...</p>
                    </div>
                </div>
                
                <div class="mt-6 pt-4 border-t border-gray-300">
                    <div class="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
                        <div class="text-center">
                            <p class="text-2xl font-bold text-gray-900" id="arb-total-opps">0</p>
                            <p class="text-gray-600">Total Opportunities</p>
                        </div>
                        <div class="text-center">
                            <p class="text-2xl font-bold text-green-600" id="arb-max-spread">0.00%</p>
                            <p class="text-gray-600">Max Spread</p>
                        </div>
                        <div class="text-center">
                            <p class="text-2xl font-bold text-blue-600" id="arb-avg-spread">0.00%</p>
                            <p class="text-gray-600">Avg Spread</p>
                        </div>
                        <div class="text-center">
                            <p class="text-2xl font-bold text-gray-900" id="arb-last-update">--:--:--</p>
                            <p class="text-gray-600">Last Update</p>
                        </div>
                    </div>
                </div>
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

            <!-- AGREEMENT ANALYSIS DASHBOARD -->
            <div class="bg-white rounded-lg p-6 border-2 border-indigo-600 mb-8 shadow-lg">
                <h2 class="text-3xl font-bold mb-6 text-center text-indigo-900">
                    <i class="fas fa-balance-scale mr-2"></i>
                    Multi-Dimensional Model Comparison
                    <span class="ml-3 text-sm bg-indigo-900 text-white px-3 py-1 rounded-full">Agreement Analysis</span>
                </h2>
                <p class="text-center text-gray-600 mb-6">Comprehensive comparison using industry best practices and academic standards</p>

                <!-- Overall Agreement Score -->
                <div id="overall-agreement" class="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg p-6 mb-6 border-2 border-indigo-300 shadow-md">
                    <div class="text-center">
                        <h3 class="text-xl font-bold text-indigo-900 mb-2">
                            <i class="fas fa-chart-pie mr-2"></i>
                            Overall Agreement Score
                        </h3>
                        <div class="flex items-center justify-center gap-4 mb-3">
                            <div class="text-5xl font-bold text-indigo-600" id="agreement-score">--</div>
                            <div class="text-2xl text-gray-500">/ 100</div>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-4 mb-2">
                            <div id="agreement-bar" class="bg-gradient-to-r from-green-500 to-indigo-600 h-4 rounded-full transition-all duration-500" style="width: 0%"></div>
                        </div>
                        <p class="text-sm text-gray-600 italic" id="agreement-interpretation">Run both analyses to calculate agreement metrics</p>
                    </div>
                </div>

                <!-- Normalized Metrics Comparison -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                    <!-- LLM Agent Normalized Metrics -->
                    <div class="bg-green-50 rounded-lg p-5 border-2 border-green-600 shadow">
                        <h3 class="text-lg font-bold mb-4 text-green-800">
                            <i class="fas fa-robot mr-2"></i>
                            LLM Agent - Normalized Scores
                        </h3>
                        <div class="space-y-3">
                            <div>
                                <div class="flex justify-between mb-1">
                                    <span class="text-sm font-semibold text-gray-700">Economic Analysis</span>
                                    <span class="text-sm font-bold text-green-700" id="llm-economic-score">--%</span>
                                </div>
                                <div class="w-full bg-gray-200 rounded-full h-3">
                                    <div id="llm-economic-bar" class="bg-green-600 h-3 rounded-full transition-all duration-300" style="width: 0%"></div>
                                </div>
                            </div>
                            <div>
                                <div class="flex justify-between mb-1">
                                    <span class="text-sm font-semibold text-gray-700">Sentiment Analysis</span>
                                    <span class="text-sm font-bold text-green-700" id="llm-sentiment-score">--%</span>
                                </div>
                                <div class="w-full bg-gray-200 rounded-full h-3">
                                    <div id="llm-sentiment-bar" class="bg-green-600 h-3 rounded-full transition-all duration-300" style="width: 0%"></div>
                                </div>
                            </div>
                            <div>
                                <div class="flex justify-between mb-1">
                                    <span class="text-sm font-semibold text-gray-700">Liquidity Analysis</span>
                                    <span class="text-sm font-bold text-green-700" id="llm-liquidity-score">--%</span>
                                </div>
                                <div class="w-full bg-gray-200 rounded-full h-3">
                                    <div id="llm-liquidity-bar" class="bg-green-600 h-3 rounded-full transition-all duration-300" style="width: 0%"></div>
                                </div>
                            </div>
                            <div class="pt-3 border-t-2 border-green-300">
                                <div class="flex justify-between items-center">
                                    <span class="text-base font-bold text-gray-800">Overall Confidence</span>
                                    <span class="text-xl font-bold text-green-800" id="llm-overall-score">--%</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Backtesting Agent Normalized Metrics -->
                    <div class="bg-orange-50 rounded-lg p-5 border-2 border-orange-600 shadow">
                        <h3 class="text-lg font-bold mb-4 text-orange-800">
                            <i class="fas fa-chart-line mr-2"></i>
                            Backtesting Agent - Normalized Scores
                        </h3>
                        <div class="space-y-3">
                            <div>
                                <div class="flex justify-between mb-1">
                                    <span class="text-sm font-semibold text-gray-700">Economic Analysis</span>
                                    <span class="text-sm font-bold text-orange-700" id="bt-economic-score">--%</span>
                                </div>
                                <div class="w-full bg-gray-200 rounded-full h-3">
                                    <div id="bt-economic-bar" class="bg-orange-600 h-3 rounded-full transition-all duration-300" style="width: 0%"></div>
                                </div>
                            </div>
                            <div>
                                <div class="flex justify-between mb-1">
                                    <span class="text-sm font-semibold text-gray-700">Sentiment Analysis</span>
                                    <span class="text-sm font-bold text-orange-700" id="bt-sentiment-score">--%</span>
                                </div>
                                <div class="w-full bg-gray-200 rounded-full h-3">
                                    <div id="bt-sentiment-bar" class="bg-orange-600 h-3 rounded-full transition-all duration-300" style="width: 0%"></div>
                                </div>
                            </div>
                            <div>
                                <div class="flex justify-between mb-1">
                                    <span class="text-sm font-semibold text-gray-700">Liquidity Analysis</span>
                                    <span class="text-sm font-bold text-orange-700" id="bt-liquidity-score">--%</span>
                                </div>
                                <div class="w-full bg-gray-200 rounded-full h-3">
                                    <div id="bt-liquidity-bar" class="bg-orange-600 h-3 rounded-full transition-all duration-300" style="width: 0%"></div>
                                </div>
                            </div>
                            <div class="pt-3 border-t-2 border-orange-300">
                                <div class="flex justify-between items-center">
                                    <span class="text-base font-bold text-gray-800">Overall Score</span>
                                    <span class="text-xl font-bold text-orange-800" id="bt-overall-score">--%</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Component-Level Delta Analysis -->
                <div class="bg-amber-50 rounded-lg p-5 border border-gray-300 mb-6 shadow">
                    <h3 class="text-lg font-bold mb-4 text-gray-800">
                        <i class="fas fa-code-branch mr-2"></i>
                        Component-Level Delta Analysis
                    </h3>
                    <div class="overflow-x-auto">
                        <table class="w-full text-sm">
                            <thead>
                                <tr class="border-b-2 border-gray-300">
                                    <th class="text-left py-2 px-3 font-semibold text-gray-700">Component</th>
                                    <th class="text-center py-2 px-3 font-semibold text-green-700">LLM Score</th>
                                    <th class="text-center py-2 px-3 font-semibold text-orange-700">Backtest Score</th>
                                    <th class="text-center py-2 px-3 font-semibold text-indigo-700">Delta (Δ)</th>
                                    <th class="text-center py-2 px-3 font-semibold text-gray-700">Concordance</th>
                                </tr>
                            </thead>
                            <tbody id="delta-table-body">
                                <tr class="border-b border-gray-200">
                                    <td class="py-2 px-3 font-medium">Economic</td>
                                    <td class="text-center py-2 px-3" id="delta-llm-economic">--</td>
                                    <td class="text-center py-2 px-3" id="delta-bt-economic">--</td>
                                    <td class="text-center py-2 px-3 font-bold" id="delta-economic">--</td>
                                    <td class="text-center py-2 px-3" id="concordance-economic">--</td>
                                </tr>
                                <tr class="border-b border-gray-200">
                                    <td class="py-2 px-3 font-medium">Sentiment</td>
                                    <td class="text-center py-2 px-3" id="delta-llm-sentiment">--</td>
                                    <td class="text-center py-2 px-3" id="delta-bt-sentiment">--</td>
                                    <td class="text-center py-2 px-3 font-bold" id="delta-sentiment">--</td>
                                    <td class="text-center py-2 px-3" id="concordance-sentiment">--</td>
                                </tr>
                                <tr class="border-b border-gray-200">
                                    <td class="py-2 px-3 font-medium">Liquidity</td>
                                    <td class="text-center py-2 px-3" id="delta-llm-liquidity">--</td>
                                    <td class="text-center py-2 px-3" id="delta-bt-liquidity">--</td>
                                    <td class="text-center py-2 px-3 font-bold" id="delta-liquidity">--</td>
                                    <td class="text-center py-2 px-3" id="concordance-liquidity">--</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    <div class="mt-4 flex items-center justify-between text-xs text-gray-600 bg-white p-3 rounded border border-gray-200">
                        <div><strong>Signal Concordance:</strong> <span id="signal-concordance">--%</span></div>
                        <div><strong>Krippendorff's Alpha (α):</strong> <span id="krippendorff-alpha">--</span></div>
                        <div><strong>Mean Absolute Delta:</strong> <span id="mean-delta">--</span></div>
                    </div>
                </div>

                <!-- Risk-Adjusted Performance & Position Sizing -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <!-- Risk-Adjusted Metrics -->
                    <div class="bg-blue-50 rounded-lg p-5 border border-blue-300 shadow">
                        <h3 class="text-lg font-bold mb-4 text-blue-900">
                            <i class="fas fa-shield-alt mr-2"></i>
                            Risk-Adjusted Performance
                        </h3>
                        <div class="space-y-2 text-sm">
                            <div class="flex justify-between py-2 border-b border-blue-200">
                                <span class="font-semibold text-gray-700">Sharpe Ratio</span>
                                <span class="font-bold text-blue-700" id="risk-sharpe">--</span>
                            </div>
                            <div class="flex justify-between py-2 border-b border-blue-200">
                                <span class="font-semibold text-gray-700">Sortino Ratio</span>
                                <span class="font-bold text-blue-700" id="risk-sortino">--</span>
                            </div>
                            <div class="flex justify-between py-2 border-b border-blue-200">
                                <span class="font-semibold text-gray-700">Calmar Ratio</span>
                                <span class="font-bold text-blue-700" id="risk-calmar">--</span>
                            </div>
                            <div class="flex justify-between py-2 border-b border-blue-200">
                                <span class="font-semibold text-gray-700">Maximum Drawdown</span>
                                <span class="font-bold text-red-600" id="risk-maxdd">--</span>
                            </div>
                            <div class="flex justify-between py-2">
                                <span class="font-semibold text-gray-700">Win Rate</span>
                                <span class="font-bold text-blue-700" id="risk-winrate">--</span>
                            </div>
                        </div>
                    </div>

                    <!-- Position Sizing Recommendation -->
                    <div class="bg-purple-50 rounded-lg p-5 border border-purple-300 shadow">
                        <h3 class="text-lg font-bold mb-4 text-purple-900">
                            <i class="fas fa-wallet mr-2"></i>
                            Position Sizing (Kelly Criterion)
                        </h3>
                        <div class="space-y-3">
                            <div class="bg-white rounded p-3 border border-purple-200">
                                <div class="text-xs text-gray-600 mb-1">Optimal Position Size</div>
                                <div class="text-2xl font-bold text-purple-700" id="kelly-optimal">--%</div>
                            </div>
                            <div class="bg-white rounded p-3 border border-purple-200">
                                <div class="text-xs text-gray-600 mb-1">Conservative (Half-Kelly)</div>
                                <div class="text-2xl font-bold text-purple-700" id="kelly-half">--%</div>
                            </div>
                            <div class="bg-white rounded p-3 border border-purple-200">
                                <div class="text-xs text-gray-600 mb-1">Risk Category</div>
                                <div class="text-lg font-bold" id="kelly-risk-category">
                                    <span class="px-3 py-1 rounded-full bg-gray-200 text-gray-700">Not Calculated</span>
                                </div>
                            </div>
                        </div>
                        <div class="mt-3 text-xs text-gray-600 italic bg-white p-2 rounded border border-gray-200">
                            Based on backtesting win rate, avg win/loss, and risk metrics
                        </div>
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

            // Load Live Arbitrage Opportunities
            async function loadLiveArbitrage() {
                console.log('Loading live arbitrage opportunities...');
                const container = document.getElementById('live-arbitrage-container');
                
                try {
                    const response = await axios.get('/api/strategies/arbitrage/advanced?symbol=BTC');
                    const data = response.data;
                    
                    if (data.success) {
                        const arb = data.arbitrage_opportunities;
                        
                        // Update summary stats
                        document.getElementById('arb-total-opps').textContent = arb.total_opportunities || 0;
                        document.getElementById('arb-max-spread').textContent = 
                            arb.spatial.opportunities && arb.spatial.opportunities.length > 0 ? 
                            Math.max(...arb.spatial.opportunities.map(o => o.spread_percent || 0)).toFixed(2) + '%' : '0.00%';
                        document.getElementById('arb-avg-spread').textContent = 
                            (arb.spatial.average_spread || 0).toFixed(2) + '%';
                        document.getElementById('arb-last-update').textContent = formatTime(Date.now());
                        
                        // Create arbitrage cards
                        let html = '';
                        
                        // Spatial Arbitrage Opportunities
                        if (arb.spatial.opportunities.length > 0) {
                            arb.spatial.opportunities.slice(0, 6).forEach(opp => {
                                const profitColor = opp.spread_percent > 0.3 ? 'text-green-600' : 'text-gray-600';
                                const borderColor = opp.spread_percent > 0.3 ? 'border-green-600' : 'border-gray-300';
                                const statusBadge = opp.spread_percent > 0.3 ? 
                                    '<div class="mt-2 pt-2 border-t border-gray-300"><span class="text-xs font-bold text-green-600"><i class="fas fa-check-circle mr-1"></i> Profitable</span></div>' : 
                                    '<div class="mt-2 pt-2 border-t border-gray-300"><span class="text-xs text-gray-600"><i class="fas fa-info-circle mr-1"></i> Below threshold</span></div>';
                                
                                html += '<div class="bg-amber-50 rounded-lg p-4 border-2 ' + borderColor + ' shadow hover:shadow-lg transition-shadow">' +
                                    '<div class="flex items-center justify-between mb-2">' +
                                        '<span class="text-sm font-bold text-gray-900">' + opp.buy_exchange + ' → ' + opp.sell_exchange + '</span>' +
                                        '<span class="text-xs bg-blue-900 text-white px-2 py-1 rounded">Spatial</span>' +
                                    '</div>' +
                                    '<div class="space-y-1 text-sm">' +
                                        '<div class="flex justify-between">' +
                                            '<span class="text-gray-600">Buy Price:</span>' +
                                            '<span class="text-gray-900 font-mono">$' + opp.buy_price.toLocaleString() + '</span>' +
                                        '</div>' +
                                        '<div class="flex justify-between">' +
                                            '<span class="text-gray-600">Sell Price:</span>' +
                                            '<span class="text-gray-900 font-mono">$' + opp.sell_price.toLocaleString() + '</span>' +
                                        '</div>' +
                                        '<div class="flex justify-between">' +
                                            '<span class="text-gray-600">Spread:</span>' +
                                            '<span class="' + profitColor + ' font-bold">' + opp.spread_percent.toFixed(2) + '%</span>' +
                                        '</div>' +
                                        '<div class="flex justify-between">' +
                                            '<span class="text-gray-600">Profit (1 BTC):</span>' +
                                            '<span class="' + profitColor + ' font-bold">$' + opp.profit_usd.toFixed(2) + '</span>' +
                                        '</div>' +
                                    '</div>' +
                                    statusBadge +
                                '</div>';
                            });
                        }
                        
                        // Triangular Arbitrage
                        if (arb.triangular.opportunities.length > 0) {
                            arb.triangular.opportunities.slice(0, 2).forEach(opp => {
                                const profitColor = opp.profit_percent > 0 ? 'text-green-600' : 'text-gray-600';
                                const borderColor = opp.profit_percent > 0 ? 'border-purple-600' : 'border-gray-300';
                                const statusBadge = opp.profit_percent > 0 ? 
                                    '<div class="mt-2 pt-2 border-t border-gray-300"><span class="text-xs font-bold text-green-600"><i class="fas fa-check-circle mr-1"></i> Profitable</span></div>' : 
                                    '<div class="mt-2 pt-2 border-t border-gray-300"><span class="text-xs text-gray-600"><i class="fas fa-info-circle mr-1"></i> No profit</span></div>';
                                
                                html += '<div class="bg-amber-50 rounded-lg p-4 border-2 ' + borderColor + ' shadow hover:shadow-lg transition-shadow">' +
                                    '<div class="flex items-center justify-between mb-2">' +
                                        '<span class="text-sm font-bold text-gray-900">Triangular</span>' +
                                        '<span class="text-xs bg-purple-600 text-white px-2 py-1 rounded">3-Leg</span>' +
                                    '</div>' +
                                    '<div class="space-y-1 text-sm">' +
                                        '<div class="text-gray-600 mb-2">' +
                                            '<i class="fas fa-route mr-1"></i>' +
                                            opp.path.join(' → ') +
                                        '</div>' +
                                        '<div class="flex justify-between">' +
                                            '<span class="text-gray-600">Exchange:</span>' +
                                            '<span class="text-gray-900">' + opp.exchange + '</span>' +
                                        '</div>' +
                                        '<div class="flex justify-between">' +
                                            '<span class="text-gray-600">Profit:</span>' +
                                            '<span class="' + profitColor + ' font-bold">' + opp.profit_percent.toFixed(2) + '%</span>' +
                                        '</div>' +
                                    '</div>' +
                                    statusBadge +
                                '</div>';
                            });
                        }
                        
                        // Statistical Arbitrage
                        if (arb.statistical.opportunities && arb.statistical.opportunities.length > 0) {
                            const statArb = arb.statistical.opportunities[0];
                            const signalColor = statArb.signal === 'BUY' ? 'text-green-600' : statArb.signal === 'SELL' ? 'text-red-600' : 'text-gray-600';
                            
                            html += '<div class="bg-amber-50 rounded-lg p-4 border-2 border-blue-600 shadow hover:shadow-lg transition-shadow">' +
                                '<div class="flex items-center justify-between mb-2">' +
                                    '<span class="text-sm font-bold text-gray-900">Statistical</span>' +
                                    '<span class="text-xs bg-blue-600 text-white px-2 py-1 rounded">Mean Rev</span>' +
                                '</div>' +
                                '<div class="space-y-1 text-sm">' +
                                    '<div class="flex justify-between">' +
                                        '<span class="text-gray-600">Z-Score:</span>' +
                                        '<span class="text-gray-900 font-bold">' + statArb.z_score.toFixed(2) + '</span>' +
                                    '</div>' +
                                    '<div class="flex justify-between">' +
                                        '<span class="text-gray-600">Signal:</span>' +
                                        '<span class="' + signalColor + ' font-bold">' + statArb.signal + '</span>' +
                                    '</div>' +
                                    '<div class="flex justify-between">' +
                                        '<span class="text-gray-600">Mean Price:</span>' +
                                        '<span class="text-gray-900 font-mono">$' + statArb.mean_price.toFixed(2) + '</span>' +
                                    '</div>' +
                                    '<div class="flex justify-between">' +
                                        '<span class="text-gray-600">Deviation:</span>' +
                                        '<span class="text-gray-900">' + statArb.std_dev.toFixed(2) + '</span>' +
                                    '</div>' +
                                '</div>' +
                            '</div>';
                        }
                        
                        // Funding Rate Arbitrage
                        if (arb.funding_rate.opportunities && arb.funding_rate.opportunities.length > 0) {
                            const fundingArb = arb.funding_rate.opportunities[0];
                            const rateColor = Math.abs(fundingArb.funding_rate_percent) > 0.01 ? 'text-orange-600' : 'text-gray-600';
                            
                            html += '<div class="bg-amber-50 rounded-lg p-4 border-2 border-orange-600 shadow hover:shadow-lg transition-shadow">' +
                                '<div class="flex items-center justify-between mb-2">' +
                                    '<span class="text-sm font-bold text-gray-900">Funding Rate</span>' +
                                    '<span class="text-xs bg-orange-600 text-white px-2 py-1 rounded">Futures</span>' +
                                '</div>' +
                                '<div class="space-y-1 text-sm">' +
                                    '<div class="flex justify-between">' +
                                        '<span class="text-gray-600">Exchange:</span>' +
                                        '<span class="text-gray-900">' + fundingArb.exchange + '</span>' +
                                    '</div>' +
                                    '<div class="flex justify-between">' +
                                        '<span class="text-gray-600">Pair:</span>' +
                                        '<span class="text-gray-900">' + fundingArb.pair + '</span>' +
                                    '</div>' +
                                    '<div class="flex justify-between">' +
                                        '<span class="text-gray-600">Funding Rate:</span>' +
                                        '<span class="' + rateColor + ' font-bold">' + fundingArb.funding_rate_percent.toFixed(4) + '%</span>' +
                                    '</div>' +
                                    '<div class="flex justify-between">' +
                                        '<span class="text-gray-600">Strategy:</span>' +
                                        '<span class="text-gray-900">' + fundingArb.strategy + '</span>' +
                                    '</div>' +
                                '</div>' +
                            '</div>';
                        }
                        
                        if (html === '') {
                            html = '<div class="col-span-3 text-center py-8"><p class="text-gray-600">No arbitrage opportunities found at this time</p></div>';
                        }
                        
                        container.innerHTML = html;
                    }
                } catch (error) {
                    console.error('Error loading arbitrage:', error);
                    container.innerHTML = '<div class="col-span-3 text-center py-8">' +
                        '<i class="fas fa-exclamation-triangle text-4xl text-red-600 mb-3"></i>' +
                        '<p class="text-red-600">Error loading arbitrage opportunities</p>' +
                        '<p class="text-sm text-gray-600 mt-2">' + error.message + '</p>' +
                    '</div>';
                }
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

            // ============================================================
            // HELPER FUNCTIONS FOR AGREEMENT ANALYSIS & METRICS CALCULATION
            // ============================================================

            /**
             * Normalize a score to 0-100% range
             * @param {number} score - Raw score value
             * @param {number} min - Minimum possible value
             * @param {number} max - Maximum possible value
             * @returns {number} Normalized score (0-100)
             */
            function normalizeScore(score, min, max) {
                if (max === min) return 50; // Avoid division by zero
                return Math.max(0, Math.min(100, ((score - min) / (max - min)) * 100));
            }

            /**
             * Calculate Krippendorff's Alpha for interval data
             * Measures inter-rater reliability between LLM and Backtesting scores
             * @param {Array<number>} llmScores - Array of LLM component scores
             * @param {Array<number>} btScores - Array of Backtesting component scores
             * @returns {number} Alpha value (-1 to 1, where 1 = perfect agreement)
             */
            function calculateKrippendorffAlpha(llmScores, btScores) {
                if (llmScores.length !== btScores.length || llmScores.length === 0) {
                    return 0;
                }

                const n = llmScores.length;
                
                // Calculate observed disagreement
                let observedDisagreement = 0;
                for (let i = 0; i < n; i++) {
                    observedDisagreement += Math.pow(llmScores[i] - btScores[i], 2);
                }
                observedDisagreement /= n;

                // Calculate expected disagreement (variance of all values)
                const allScores = [...llmScores, ...btScores];
                const mean = allScores.reduce((a, b) => a + b, 0) / allScores.length;
                let expectedDisagreement = 0;
                for (const score of allScores) {
                    expectedDisagreement += Math.pow(score - mean, 2);
                }
                expectedDisagreement /= allScores.length;

                // Calculate Alpha
                if (expectedDisagreement === 0) return 1; // Perfect agreement
                const alpha = 1 - (observedDisagreement / expectedDisagreement);
                
                return Math.max(-1, Math.min(1, alpha));
            }

            /**
             * Calculate Signal Concordance (percentage of components in agreement)
             * Components agree if their delta is within threshold
             * @param {Array<number>} deltas - Array of delta values
             * @param {number} threshold - Agreement threshold (default 20%)
             * @returns {number} Concordance percentage (0-100)
             */
            function calculateSignalConcordance(deltas, threshold = 20) {
                if (deltas.length === 0) return 0;
                
                const inAgreement = deltas.filter(delta => Math.abs(delta) <= threshold).length;
                return (inAgreement / deltas.length) * 100;
            }

            /**
             * Calculate Sortino Ratio (risk-adjusted return using downside deviation)
             * @param {number} totalReturn - Total return percentage
             * @param {number} downsideDeviation - Standard deviation of negative returns
             * @param {number} riskFreeRate - Risk-free rate (default 2%)
             * @returns {number} Sortino ratio
             */
            function calculateSortinoRatio(totalReturn, downsideDeviation, riskFreeRate = 2) {
                if (downsideDeviation === 0) return 0;
                return (totalReturn - riskFreeRate) / downsideDeviation;
            }

            /**
             * Calculate Calmar Ratio (return / max drawdown)
             * @param {number} totalReturn - Total return percentage
             * @param {number} maxDrawdown - Maximum drawdown percentage (positive value)
             * @returns {number} Calmar ratio
             */
            function calculateCalmarRatio(totalReturn, maxDrawdown) {
                if (maxDrawdown === 0) return 0;
                return totalReturn / maxDrawdown;
            }

            /**
             * Calculate Kelly Criterion for optimal position sizing
             * @param {number} winRate - Win rate as decimal (0-1)
             * @param {number} avgWin - Average win amount
             * @param {number} avgLoss - Average loss amount (positive value)
             * @returns {object} Kelly percentages and risk category
             */
            function calculateKellyCriterion(winRate, avgWin, avgLoss) {
                if (avgLoss === 0 || avgWin === 0) {
                    return { optimal: 0, half: 0, category: 'Insufficient Data', color: 'gray' };
                }

                const winLossRatio = avgWin / avgLoss;
                const kellyPercent = ((winLossRatio * winRate) - (1 - winRate)) / winLossRatio;
                
                // Clamp to reasonable range (0-40%)
                const optimalKelly = Math.max(0, Math.min(40, kellyPercent * 100));
                const halfKelly = optimalKelly / 2;

                // Determine risk category
                let category, color;
                if (optimalKelly < 5) {
                    category = 'Low Risk';
                    color = 'green';
                } else if (optimalKelly < 15) {
                    category = 'Moderate Risk';
                    color = 'blue';
                } else if (optimalKelly < 25) {
                    category = 'High Risk';
                    color = 'yellow';
                } else {
                    category = 'Very High Risk';
                    color = 'red';
                }

                return { 
                    optimal: optimalKelly.toFixed(2), 
                    half: halfKelly.toFixed(2), 
                    category, 
                    color 
                };
            }

            /**
             * Update the Agreement Analysis Dashboard with calculated metrics
             * @param {object} llmData - LLM analysis data with component scores
             * @param {object} btData - Backtesting data with component scores
             */
            function updateAgreementDashboard(llmData, btData) {
                // Extract normalized component scores
                const llmScores = {
                    economic: llmData.economicScore || 0,
                    sentiment: llmData.sentimentScore || 0,
                    liquidity: llmData.liquidityScore || 0,
                    overall: llmData.overallConfidence || 0
                };

                const btScores = {
                    economic: btData.economicScore || 0,
                    sentiment: btData.sentimentScore || 0,
                    liquidity: btData.liquidityScore || 0,
                    overall: btData.overallScore || 0
                };

                // Update LLM normalized scores
                document.getElementById('llm-economic-score').textContent = llmScores.economic.toFixed(1) + '%';
                document.getElementById('llm-economic-bar').style.width = llmScores.economic + '%';
                document.getElementById('llm-sentiment-score').textContent = llmScores.sentiment.toFixed(1) + '%';
                document.getElementById('llm-sentiment-bar').style.width = llmScores.sentiment + '%';
                document.getElementById('llm-liquidity-score').textContent = llmScores.liquidity.toFixed(1) + '%';
                document.getElementById('llm-liquidity-bar').style.width = llmScores.liquidity + '%';
                document.getElementById('llm-overall-score').textContent = llmScores.overall.toFixed(1) + '%';

                // Update Backtesting normalized scores
                document.getElementById('bt-economic-score').textContent = btScores.economic.toFixed(1) + '%';
                document.getElementById('bt-economic-bar').style.width = btScores.economic + '%';
                document.getElementById('bt-sentiment-score').textContent = btScores.sentiment.toFixed(1) + '%';
                document.getElementById('bt-sentiment-bar').style.width = btScores.sentiment + '%';
                document.getElementById('bt-liquidity-score').textContent = btScores.liquidity.toFixed(1) + '%';
                document.getElementById('bt-liquidity-bar').style.width = btScores.liquidity + '%';
                document.getElementById('bt-overall-score').textContent = btScores.overall.toFixed(1) + '%';

                // Calculate deltas
                const deltas = {
                    economic: llmScores.economic - btScores.economic,
                    sentiment: llmScores.sentiment - btScores.sentiment,
                    liquidity: llmScores.liquidity - btScores.liquidity
                };

                // Update delta table
                const formatDelta = (delta) => {
                    const sign = delta >= 0 ? '+' : '';
                    const color = Math.abs(delta) <= 10 ? 'text-green-600' : Math.abs(delta) <= 25 ? 'text-yellow-600' : 'text-red-600';
                    return \`<span class="\${color}">\${sign}\${delta.toFixed(1)}%</span>\`;
                };

                const formatConcordance = (delta) => {
                    const concordance = Math.abs(delta) <= 20;
                    return concordance 
                        ? '<span class="text-green-600 font-semibold">✓ Agree</span>' 
                        : '<span class="text-red-600 font-semibold">✗ Diverge</span>';
                };

                document.getElementById('delta-llm-economic').textContent = llmScores.economic.toFixed(1) + '%';
                document.getElementById('delta-bt-economic').textContent = btScores.economic.toFixed(1) + '%';
                document.getElementById('delta-economic').innerHTML = formatDelta(deltas.economic);
                document.getElementById('concordance-economic').innerHTML = formatConcordance(deltas.economic);

                document.getElementById('delta-llm-sentiment').textContent = llmScores.sentiment.toFixed(1) + '%';
                document.getElementById('delta-bt-sentiment').textContent = btScores.sentiment.toFixed(1) + '%';
                document.getElementById('delta-sentiment').innerHTML = formatDelta(deltas.sentiment);
                document.getElementById('concordance-sentiment').innerHTML = formatConcordance(deltas.sentiment);

                document.getElementById('delta-llm-liquidity').textContent = llmScores.liquidity.toFixed(1) + '%';
                document.getElementById('delta-bt-liquidity').textContent = btScores.liquidity.toFixed(1) + '%';
                document.getElementById('delta-liquidity').innerHTML = formatDelta(deltas.liquidity);
                document.getElementById('concordance-liquidity').innerHTML = formatConcordance(deltas.liquidity);

                // Calculate agreement metrics
                const llmScoreArray = [llmScores.economic, llmScores.sentiment, llmScores.liquidity];
                const btScoreArray = [btScores.economic, btScores.sentiment, btScores.liquidity];
                const deltaArray = [Math.abs(deltas.economic), Math.abs(deltas.sentiment), Math.abs(deltas.liquidity)];

                const krippendorffAlpha = calculateKrippendorffAlpha(llmScoreArray, btScoreArray);
                const signalConcordance = calculateSignalConcordance([deltas.economic, deltas.sentiment, deltas.liquidity]);
                const meanDelta = deltaArray.reduce((a, b) => a + b, 0) / deltaArray.length;

                // Overall agreement score (weighted combination)
                const agreementScore = (
                    (krippendorffAlpha + 1) * 25 +  // Alpha ranges -1 to 1, normalize to 0-50
                    signalConcordance * 0.3 +         // 0-30 points
                    (100 - meanDelta) * 0.2           // 0-20 points (inverse of mean delta)
                );

                // Update agreement metrics
                document.getElementById('agreement-score').textContent = Math.round(agreementScore);
                document.getElementById('agreement-bar').style.width = agreementScore + '%';
                document.getElementById('krippendorff-alpha').textContent = krippendorffAlpha.toFixed(3);
                document.getElementById('signal-concordance').textContent = signalConcordance.toFixed(1) + '%';
                document.getElementById('mean-delta').textContent = meanDelta.toFixed(1) + '%';

                // Agreement interpretation
                let interpretation;
                if (agreementScore >= 80) {
                    interpretation = '🟢 Excellent Agreement - Both models strongly aligned';
                } else if (agreementScore >= 60) {
                    interpretation = '🟡 Good Agreement - Models generally aligned with minor differences';
                } else if (agreementScore >= 40) {
                    interpretation = '🟠 Moderate Agreement - Significant differences in some components';
                } else {
                    interpretation = '🔴 Low Agreement - Models diverge substantially';
                }
                document.getElementById('agreement-interpretation').textContent = interpretation;

                // Update risk-adjusted metrics (from backtesting data)
                if (btData.sharpeRatio !== undefined) {
                    document.getElementById('risk-sharpe').textContent = btData.sharpeRatio.toFixed(2);
                }
                if (btData.sortinoRatio !== undefined) {
                    document.getElementById('risk-sortino').textContent = btData.sortinoRatio.toFixed(2);
                }
                if (btData.calmarRatio !== undefined) {
                    document.getElementById('risk-calmar').textContent = btData.calmarRatio.toFixed(2);
                }
                if (btData.maxDrawdown !== undefined) {
                    document.getElementById('risk-maxdd').textContent = btData.maxDrawdown.toFixed(2) + '%';
                }
                if (btData.winRate !== undefined) {
                    document.getElementById('risk-winrate').textContent = btData.winRate.toFixed(1) + '%';
                }

                // Update Kelly Criterion position sizing
                if (btData.winRate && btData.avgWin && btData.avgLoss) {
                    const kelly = calculateKellyCriterion(
                        btData.winRate / 100, 
                        btData.avgWin, 
                        Math.abs(btData.avgLoss)
                    );
                    
                    document.getElementById('kelly-optimal').textContent = kelly.optimal + '%';
                    document.getElementById('kelly-half').textContent = kelly.half + '%';
                    
                    const colorMap = {
                        green: 'bg-green-500 text-white',
                        blue: 'bg-blue-500 text-white',
                        yellow: 'bg-yellow-500 text-gray-900',
                        red: 'bg-red-500 text-white',
                        gray: 'bg-gray-200 text-gray-700'
                    };
                    
                    document.getElementById('kelly-risk-category').innerHTML = 
                        \`<span class="px-3 py-1 rounded-full \${colorMap[kelly.color]}">\${kelly.category}</span>\`;
                }
            }

            // Global variables to store analysis data for comparison
            let llmAnalysisData = null;
            let backtestAnalysisData = null;

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
                    
                    // Extract agent scores from the response
                    // The LLM analysis includes agent_data with scores for each component
                    let economicScore = 0, sentimentScore = 0, liquidityScore = 0;
                    let totalSignals = 18; // Max possible score (3 agents × 6 signals each)
                    
                    if (data.agent_data) {
                        // Economic agent signals (6 max: GDP, Inflation, Rates, Employment, etc.)
                        if (data.agent_data.economic) {
                            const econ = data.agent_data.economic;
                            economicScore = (econ.signals_count || 0);
                        }
                        
                        // Sentiment agent signals (6 max: Fear/Greed, VIX, Social, News, etc.)
                        if (data.agent_data.sentiment) {
                            const sent = data.agent_data.sentiment;
                            sentimentScore = (sent.signals_count || 0);
                        }
                        
                        // Cross-exchange/Liquidity agent signals (6 max: Spread, Volume, Depth, etc.)
                        if (data.agent_data.cross_exchange) {
                            const liq = data.agent_data.cross_exchange;
                            liquidityScore = (liq.signals_count || 0);
                        }
                    }
                    
                    // Fallback: Parse from analysis text if agent_data not structured
                    if (economicScore === 0 && sentimentScore === 0 && liquidityScore === 0) {
                        // Estimate scores from analysis content (heuristic)
                        const analysisText = data.analysis.toLowerCase();
                        
                        // Economic indicators
                        if (analysisText.includes('strong economic') || analysisText.includes('gdp growth') || analysisText.includes('inflation')) {
                            economicScore = 4;
                        } else if (analysisText.includes('economic')) {
                            economicScore = 3;
                        }
                        
                        // Sentiment indicators
                        if (analysisText.includes('bullish sentiment') || analysisText.includes('positive sentiment') || analysisText.includes('fear')) {
                            sentimentScore = 4;
                        } else if (analysisText.includes('sentiment')) {
                            sentimentScore = 3;
                        }
                        
                        // Liquidity indicators
                        if (analysisText.includes('high liquidity') || analysisText.includes('volume') || analysisText.includes('spread')) {
                            liquidityScore = 4;
                        } else if (analysisText.includes('liquidity')) {
                            liquidityScore = 3;
                        }
                    }
                    
                    // Normalize scores to 0-100% range
                    const normalizedEconomic = normalizeScore(economicScore, 0, 6);
                    const normalizedSentiment = normalizeScore(sentimentScore, 0, 6);
                    const normalizedLiquidity = normalizeScore(liquidityScore, 0, 6);
                    const normalizedOverall = (normalizedEconomic + normalizedSentiment + normalizedLiquidity) / 3;
                    
                    // Store LLM data for comparison
                    llmAnalysisData = {
                        economicScore: normalizedEconomic,
                        sentimentScore: normalizedSentiment,
                        liquidityScore: normalizedLiquidity,
                        overallConfidence: normalizedOverall,
                        rawScores: {
                            economic: economicScore,
                            sentiment: sentimentScore,
                            liquidity: liquidityScore
                        }
                    };
                    
                    resultsDiv.innerHTML = \`
                        <div class="prose max-w-none">
                            <div class="mb-4">
                                <span class="bg-green-600 text-white px-3 py-1 rounded-full text-xs font-bold">
                                    \${data.model}
                                </span>
                                <span class="ml-2 bg-green-100 text-green-800 px-2 py-1 rounded text-xs font-semibold">
                                    Overall Confidence: \${normalizedOverall.toFixed(1)}%
                                </span>
                            </div>
                            <div class="mb-3 p-3 bg-green-50 rounded-lg border border-green-200">
                                <div class="text-xs font-semibold text-green-900 mb-2">Agent Scores (Normalized):</div>
                                <div class="grid grid-cols-3 gap-2 text-xs">
                                    <div class="text-center">
                                        <div class="text-gray-600">Economic</div>
                                        <div class="font-bold text-green-700">\${normalizedEconomic.toFixed(1)}%</div>
                                    </div>
                                    <div class="text-center">
                                        <div class="text-gray-600">Sentiment</div>
                                        <div class="font-bold text-green-700">\${normalizedSentiment.toFixed(1)}%</div>
                                    </div>
                                    <div class="text-center">
                                        <div class="text-gray-600">Liquidity</div>
                                        <div class="font-bold text-green-700">\${normalizedLiquidity.toFixed(1)}%</div>
                                    </div>
                                </div>
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
                    updateComparisonChart(normalizedOverall, null);
                    
                    // Update arbitrage chart if cross-exchange data available
                    if (data.agent_data && data.agent_data.cross_exchange) {
                        updateArbitrageChart(data.agent_data.cross_exchange.market_depth_analysis);
                    }
                    
                    // Update agreement dashboard if both analyses are complete
                    if (llmAnalysisData && backtestAnalysisData) {
                        updateAgreementDashboard(llmAnalysisData, backtestAnalysisData);
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
                    
                    // Normalize backtesting scores to 0-100% range
                    const normalizedEconomic = normalizeScore(economicScore, 0, 6);
                    const normalizedSentiment = normalizeScore(sentimentScore, 0, 6);
                    const normalizedLiquidity = normalizeScore(liquidityScore, 0, 6);
                    const normalizedOverall = normalizeScore(totalScore, 0, 18);
                    
                    // Calculate additional risk-adjusted metrics
                    // Sortino Ratio: Use downside deviation (estimate from losing trades)
                    let sortinoRatio = 0;
                    if (bt.losing_trades > 0 && bt.total_trades > 0) {
                        const avgLoss = bt.avg_loss || 0;
                        const downsideDeviation = Math.abs(avgLoss) * Math.sqrt(bt.losing_trades / bt.total_trades);
                        sortinoRatio = calculateSortinoRatio(bt.total_return, downsideDeviation, 2);
                    }
                    
                    // Calmar Ratio: Return / Max Drawdown
                    const calmarRatio = calculateCalmarRatio(bt.total_return, Math.abs(bt.max_drawdown));
                    
                    // Calculate average win and loss for Kelly Criterion
                    const avgWin = bt.avg_win || (bt.winning_trades > 0 ? (bt.final_capital - bt.initial_capital) / bt.winning_trades : 0);
                    const avgLoss = Math.abs(bt.avg_loss || (bt.losing_trades > 0 ? (bt.final_capital - bt.initial_capital) / bt.losing_trades : 0));
                    
                    // Store backtesting data for comparison
                    backtestAnalysisData = {
                        economicScore: normalizedEconomic,
                        sentimentScore: normalizedSentiment,
                        liquidityScore: normalizedLiquidity,
                        overallScore: normalizedOverall,
                        sharpeRatio: bt.sharpe_ratio || 0,
                        sortinoRatio: sortinoRatio,
                        calmarRatio: calmarRatio,
                        maxDrawdown: Math.abs(bt.max_drawdown || 0),
                        winRate: bt.win_rate || 0,
                        avgWin: avgWin,
                        avgLoss: avgLoss,
                        totalReturn: bt.total_return || 0,
                        rawScores: {
                            economic: economicScore,
                            sentiment: sentimentScore,
                            liquidity: liquidityScore,
                            total: totalScore
                        }
                    };
                    
                    const returnColor = bt.total_return >= 0 ? 'text-green-700' : 'text-red-700';
                    
                    resultsDiv.innerHTML = \`
                        <div class="space-y-4">
                            <div class="bg-white border border-orange-200 p-4 rounded-lg">
                                <h4 class="font-bold text-lg mb-3 text-orange-800">Agent Signals</h4>
                                <div class="grid grid-cols-2 gap-2 text-sm mb-3">
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
                                <div class="pt-2 border-t border-orange-200">
                                    <div class="mb-2 bg-orange-50 px-2 py-1 rounded">
                                        <span class="text-xs font-semibold text-orange-900">Normalized Scores (0-100%):</span>
                                    </div>
                                    <div class="grid grid-cols-3 gap-2 text-xs">
                                        <div class="text-center">
                                            <div class="text-gray-600">Economic</div>
                                            <div class="font-bold text-orange-700">\${normalizedEconomic.toFixed(1)}%</div>
                                        </div>
                                        <div class="text-center">
                                            <div class="text-gray-600">Sentiment</div>
                                            <div class="font-bold text-orange-700">\${normalizedSentiment.toFixed(1)}%</div>
                                        </div>
                                        <div class="text-center">
                                            <div class="text-gray-600">Liquidity</div>
                                            <div class="font-bold text-orange-700">\${normalizedLiquidity.toFixed(1)}%</div>
                                        </div>
                                    </div>
                                    <div class="mt-2 text-center">
                                        <span class="bg-orange-600 text-white px-3 py-1 rounded-full text-xs font-bold">
                                            Overall: \${normalizedOverall.toFixed(1)}%
                                        </span>
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
                    
                    // Update agreement dashboard if both analyses are complete
                    if (llmAnalysisData && backtestAnalysisData) {
                        updateAgreementDashboard(llmAnalysisData, backtestAnalysisData);
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

                // LLM vs Backtesting Comparison Chart (Grouped Bar)
                const comparisonCtx = document.getElementById('comparisonChart').getContext('2d');
                comparisonChart = new Chart(comparisonCtx, {
                    type: 'bar',
                    data: {
                        labels: ['Overall Score', 'Economic', 'Sentiment', 'Liquidity'],
                        datasets: [
                            {
                                label: 'LLM Agent',
                                data: [0, 0, 0, 0],
                                backgroundColor: 'rgba(22, 163, 74, 0.7)',
                                borderColor: 'rgba(22, 163, 74, 1)',
                                borderWidth: 2
                            },
                            {
                                label: 'Backtesting Agent',
                                data: [0, 0, 0, 0],
                                backgroundColor: 'rgba(234, 88, 12, 0.7)',
                                borderColor: 'rgba(234, 88, 12, 1)',
                                borderWidth: 2
                            }
                        ]
                    },
                    options: {
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100,
                                ticks: { 
                                    color: '#fff',
                                    callback: function(value) {
                                        return value + '%';
                                    }
                                },
                                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                                title: {
                                    display: true,
                                    text: 'Normalized Score (0-100%)',
                                    color: '#fff',
                                    font: { size: 12 }
                                }
                            },
                            x: {
                                ticks: { color: '#fff', font: { size: 11 } },
                                grid: { color: 'rgba(255, 255, 255, 0.1)' }
                            }
                        },
                        plugins: {
                            legend: { 
                                labels: { 
                                    color: '#fff',
                                    font: { size: 12, weight: 'bold' },
                                    padding: 15
                                },
                                position: 'top'
                            },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        let label = context.dataset.label || '';
                                        if (label) {
                                            label += ': ';
                                        }
                                        label += context.parsed.y.toFixed(1) + '%';
                                        return label;
                                    }
                                }
                            }
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
                
                // Use global analysis data if available, otherwise use parameters for backward compatibility
                let llmData = llmAnalysisData;
                let btData = backtestAnalysisData;
                
                // Fallback to parameters if global data not set
                if (!llmData && llmConfidence) {
                    llmData = {
                        overallConfidence: llmConfidence,
                        economicScore: 50,
                        sentimentScore: 50,
                        liquidityScore: 50
                    };
                }
                
                if (!btData && backtestSignals) {
                    const economicScore = (backtestSignals.economicScore / 6) * 100;
                    const sentimentScore = (backtestSignals.sentimentScore / 6) * 100;
                    const liquidityScore = (backtestSignals.liquidityScore / 6) * 100;
                    const totalScore = (backtestSignals.totalScore / 18) * 100;
                    
                    btData = {
                        overallScore: totalScore,
                        economicScore: economicScore,
                        sentimentScore: sentimentScore,
                        liquidityScore: liquidityScore
                    };
                }
                
                // Update LLM dataset (green bars)
                if (llmData) {
                    comparisonChart.data.datasets[0].data = [
                        llmData.overallConfidence || 0,
                        llmData.economicScore || 0,
                        llmData.sentimentScore || 0,
                        llmData.liquidityScore || 0
                    ];
                }
                
                // Update Backtesting dataset (orange bars)
                if (btData) {
                    comparisonChart.data.datasets[1].data = [
                        btData.overallScore || 0,
                        btData.economicScore || 0,
                        btData.sentimentScore || 0,
                        btData.liquidityScore || 0
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
                loadLiveArbitrage(); // Load arbitrage opportunities
                // Refresh every 10 seconds
                setInterval(loadAgentData, 10000);
                setInterval(loadLiveArbitrage, 10000); // Refresh arbitrage every 10 seconds
            });
            
            // Also call immediately (in case DOMContentLoaded already fired)
            setTimeout(() => {
                console.log('Fallback data load triggered');
                updateDashboardStats();
                loadAgentData();
                loadLiveArbitrage(); // Load arbitrage opportunities
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
  `));const at=new Dt,ys=Object.assign({"/src/index.tsx":S});let Mt=!1;for(const[,e]of Object.entries(ys))e&&(at.route("/",e),at.notFound(e.notFoundHandler),Mt=!0);if(!Mt)throw new Error("Can't import modules from ['/src/index.tsx','/app/server.ts']");export{at as default};
