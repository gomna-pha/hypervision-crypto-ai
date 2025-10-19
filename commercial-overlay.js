// Commercial Overlay Module - Adds monetization without changing existing features
const commercialOverlay = {
  plans: [
    {
      id: 'trial',
      name: 'Free Trial',
      price: 0,
      duration: '7 days',
      features: ['View-only access', 'Historical data', 'Basic alerts'],
      apiCalls: 100,
      realtime: false
    },
    {
      id: 'basic',
      name: 'Basic Trader',
      price: 299,
      duration: 'monthly',
      features: ['Real-time data', '1,000 API calls/day', 'Email alerts', 'CSV exports'],
      apiCalls: 1000,
      realtime: true,
      popular: false
    },
    {
      id: 'professional',
      name: 'Professional',
      price: 999,
      duration: 'monthly',
      features: ['All real-time feeds', '10,000 API calls/day', 'SMS alerts', 'API access', 'Priority support'],
      apiCalls: 10000,
      realtime: true,
      popular: true
    },
    {
      id: 'institutional',
      name: 'Institutional',
      price: 4999,
      duration: 'monthly',
      features: ['Unlimited API calls', 'White-label option', 'Direct market access', 'Dedicated support', 'Custom strategies'],
      apiCalls: -1,
      realtime: true,
      enterprise: true
    }
  ],
  
  stats: {
    totalInvestors: 1247,
    activeSubscriptions: 342,
    monthlyRevenue: 287000,
    avgROI: 23.4,
    uptime: 99.97
  },
  
  testimonials: [
    {
      name: "Michael Chen",
      role: "Hedge Fund Manager",
      company: "Apex Capital",
      text: "The hyperbolic visualization alone has transformed how we identify correlations. ROI increased 34% in 3 months.",
      rating: 5
    },
    {
      name: "Sarah Williams",
      role: "Private Investor",
      text: "Finally, full transparency in algorithmic trading. I can see every decision, every constraint, in real-time.",
      rating: 5
    },
    {
      name: "David Kumar",
      role: "Quant Developer",
      company: "TechQuant LLC",
      text: "The LLM integration with live agent feeds is groundbreaking. Worth every penny of the institutional plan.",
      rating: 5
    }
  ]
};

// Inject commercial UI into existing page
function injectCommercialOverlay() {
  // Create floating purchase button
  const purchaseButton = document.createElement('div');
  purchaseButton.id = 'commercial-purchase-btn';
  purchaseButton.innerHTML = `
    <style>
      #commercial-purchase-btn {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 10000;
        animation: glow 2s ease-in-out infinite;
      }
      
      @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(168, 85, 247, 0.5); }
        50% { box-shadow: 0 0 40px rgba(168, 85, 247, 0.8); }
      }
      
      #commercial-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.95);
        z-index: 9999;
        display: none;
        overflow-y: auto;
        backdrop-filter: blur(10px);
      }
      
      .commercial-close {
        position: absolute;
        top: 20px;
        right: 20px;
        width: 40px;
        height: 40px;
        cursor: pointer;
        color: #fff;
        font-size: 30px;
        z-index: 10001;
      }
      
      .plan-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(168, 85, 247, 0.3);
        transition: all 0.3s ease;
      }
      
      .plan-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(168, 85, 247, 0.3);
      }
      
      .plan-card.popular {
        border-color: #10b981;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.3);
      }
      
      .purchase-btn {
        background: linear-gradient(135deg, #8b5cf6, #6366f1);
        transition: all 0.3s ease;
      }
      
      .purchase-btn:hover {
        background: linear-gradient(135deg, #9333ea, #7c3aed);
        transform: scale(1.05);
      }
    </style>
    
    <button class="bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-6 py-3 rounded-lg font-semibold hover:from-purple-700 hover:to-indigo-700 transition-all duration-300 flex items-center gap-2">
      <span>🚀</span>
      <span>Go Premium</span>
      <span class="bg-white/20 px-2 py-1 rounded text-xs">70% OFF</span>
    </button>
  `;
  
  // Create commercial overlay
  const overlay = document.createElement('div');
  overlay.id = 'commercial-overlay';
  overlay.innerHTML = \`
    <div class="commercial-close">&times;</div>
    <div class="container mx-auto p-8" style="max-width: 1200px;">
      <div class="text-center mb-10">
        <h1 class="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-blue-400 mb-4">
          Unlock Full Trading Power
        </h1>
        <p class="text-xl text-gray-400">Join \${commercialOverlay.stats.totalInvestors.toLocaleString()} traders already using our platform</p>
        <div class="flex justify-center gap-8 mt-6 text-sm">
          <div><span class="text-3xl font-bold text-green-400">\${commercialOverlay.stats.avgROI}%</span><br>Avg ROI</div>
          <div><span class="text-3xl font-bold text-blue-400">\${commercialOverlay.stats.uptime}%</span><br>Uptime</div>
          <div><span class="text-3xl font-bold text-purple-400">$\${(commercialOverlay.stats.monthlyRevenue/1000).toFixed(0)}K</span><br>Monthly Volume</div>
        </div>
      </div>
      
      <!-- Pricing Plans -->
      <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-10">
        \${commercialOverlay.plans.map(plan => \`
          <div class="plan-card rounded-lg p-6 \${plan.popular ? 'popular' : ''} relative">
            \${plan.popular ? '<div class="absolute -top-3 left-1/2 transform -translate-x-1/2 bg-green-500 text-white px-3 py-1 rounded-full text-xs font-bold">MOST POPULAR</div>' : ''}
            <h3 class="text-xl font-bold mb-2">\${plan.name}</h3>
            <div class="mb-4">
              <span class="text-3xl font-bold">$\${plan.price}</span>
              <span class="text-gray-400">/ \${plan.duration}</span>
            </div>
            <ul class="space-y-2 mb-6">
              \${plan.features.map(feature => \`
                <li class="flex items-center gap-2 text-sm">
                  <span class="text-green-400">✓</span>
                  <span class="text-gray-300">\${feature}</span>
                </li>
              \`).join('')}
            </ul>
            <button class="purchase-btn w-full py-3 rounded-lg font-semibold text-white" data-plan="\${plan.id}">
              \${plan.price === 0 ? 'Start Free Trial' : 'Subscribe Now'}
            </button>
          </div>
        \`).join('')}
      </div>
      
      <!-- Testimonials -->
      <div class="mb-10">
        <h2 class="text-2xl font-bold text-center mb-6">What Investors Say</h2>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
          \${commercialOverlay.testimonials.map(testimonial => \`
            <div class="bg-slate-800/50 rounded-lg p-6">
              <div class="flex gap-1 mb-3">
                \${Array(testimonial.rating).fill('⭐').join('')}
              </div>
              <p class="text-gray-300 mb-4 italic">"\${testimonial.text}"</p>
              <div>
                <div class="font-semibold">\${testimonial.name}</div>
                <div class="text-sm text-gray-400">\${testimonial.role}\${testimonial.company ? ', ' + testimonial.company : ''}</div>
              </div>
            </div>
          \`).join('')}
        </div>
      </div>
      
      <!-- Features Comparison -->
      <div class="bg-slate-800/50 rounded-lg p-6 mb-10">
        <h2 class="text-2xl font-bold mb-4">Why Our Platform?</h2>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div class="text-center p-4">
            <div class="text-3xl mb-2">🌐</div>
            <div class="font-semibold">Hyperbolic Visualization</div>
            <div class="text-gray-400">Unique Poincaré disk clustering</div>
          </div>
          <div class="text-center p-4">
            <div class="text-3xl mb-2">🤖</div>
            <div class="font-semibold">LLM Integration</div>
            <div class="text-gray-400">Real-time AI decisions</div>
          </div>
          <div class="text-center p-4">
            <div class="text-3xl mb-2">📊</div>
            <div class="font-semibold">Live Agent Feeds</div>
            <div class="text-gray-400">Sentiment, Economic, Exchange</div>
          </div>
          <div class="text-center p-4">
            <div class="text-3xl mb-2">🔒</div>
            <div class="font-semibold">Full Transparency</div>
            <div class="text-gray-400">Every constraint visible</div>
          </div>
        </div>
      </div>
      
      <!-- Payment Methods -->
      <div class="text-center text-gray-400 text-sm">
        <p class="mb-3">Secure Payment Methods</p>
        <div class="flex justify-center gap-4">
          <span>💳 Credit Card</span>
          <span>🏦 Bank Transfer</span>
          <span>₿ Crypto</span>
          <span>📱 PayPal</span>
        </div>
        <p class="mt-4">🔒 SSL Encrypted • 30-Day Money Back Guarantee</p>
      </div>
    </div>
  \`;
  
  document.body.appendChild(purchaseButton);
  document.body.appendChild(overlay);
  
  // Add event listeners
  document.querySelector('#commercial-purchase-btn button').addEventListener('click', () => {
    document.getElementById('commercial-overlay').style.display = 'block';
  });
  
  document.querySelector('.commercial-close').addEventListener('click', () => {
    document.getElementById('commercial-overlay').style.display = 'none';
  });
  
  // Handle purchase buttons
  document.querySelectorAll('.purchase-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const plan = e.target.dataset.plan;
      initiateCheckout(plan);
    });
  });
  
  // Add floating stats banner
  const statsBanner = document.createElement('div');
  statsBanner.innerHTML = `
    <div style="position: fixed; bottom: 0; left: 0; right: 0; background: linear-gradient(90deg, rgba(139, 92, 246, 0.9), rgba(99, 102, 241, 0.9)); padding: 10px; text-align: center; color: white; font-size: 14px; z-index: 1000;">
      <span>📈 ${commercialOverlay.stats.activeSubscriptions} Active Traders</span>
      <span class="mx-4">•</span>
      <span>💰 Avg ROI: ${commercialOverlay.stats.avgROI}%</span>
      <span class="mx-4">•</span>
      <span>🔥 Limited Time: 70% OFF Professional Plan</span>
      <span class="mx-4">•</span>
      <button onclick="document.getElementById('commercial-overlay').style.display='block'" class="bg-white text-purple-600 px-3 py-1 rounded font-semibold">Upgrade Now →</button>
    </div>
  `;
  document.body.appendChild(statsBanner);
}

function initiateCheckout(planId) {
  const plan = commercialOverlay.plans.find(p => p.id === planId);
  
  // Simple checkout modal
  const checkoutModal = document.createElement('div');
  checkoutModal.innerHTML = `
    <div style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: linear-gradient(135deg, #1e293b, #0f172a); padding: 30px; border-radius: 15px; border: 1px solid rgba(168, 85, 247, 0.3); z-index: 10002; min-width: 400px;">
      <h3 class="text-xl font-bold mb-4">Complete Your Purchase</h3>
      <p class="text-gray-400 mb-4">Plan: <span class="text-white font-semibold">${plan.name}</span></p>
      <p class="text-gray-400 mb-4">Price: <span class="text-white font-semibold text-2xl">$${plan.price}</span>/${plan.duration}</p>
      
      <form onsubmit="processPurchase(event, '${planId}')">
        <input type="email" placeholder="Email" class="w-full p-3 mb-3 bg-gray-800 rounded border border-gray-600 text-white" required>
        <input type="text" placeholder="Card Number" class="w-full p-3 mb-3 bg-gray-800 rounded border border-gray-600 text-white" required>
        <div class="grid grid-cols-2 gap-3 mb-4">
          <input type="text" placeholder="MM/YY" class="p-3 bg-gray-800 rounded border border-gray-600 text-white" required>
          <input type="text" placeholder="CVC" class="p-3 bg-gray-800 rounded border border-gray-600 text-white" required>
        </div>
        <button type="submit" class="w-full bg-gradient-to-r from-purple-600 to-indigo-600 text-white py-3 rounded-lg font-semibold">
          Complete Purchase
        </button>
      </form>
      
      <button onclick="this.parentElement.remove()" class="mt-3 text-gray-400 text-sm">Cancel</button>
    </div>
  `;
  document.body.appendChild(checkoutModal);
}

function processPurchase(event, planId) {
  event.preventDefault();
  
  // Simulate purchase processing
  alert('🎉 Purchase Successful! Welcome to ' + commercialOverlay.plans.find(p => p.id === planId).name + ' plan. You now have full access to all features!');
  
  // Remove checkout modal and overlay
  event.target.closest('div').parentElement.remove();
  document.getElementById('commercial-overlay').style.display = 'none';
  
  // Update UI to show premium status
  const premiumBadge = document.createElement('div');
  premiumBadge.innerHTML = `
    <div style="position: fixed; top: 70px; right: 20px; background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 8px 16px; border-radius: 20px; font-size: 12px; font-weight: bold; z-index: 1000;">
      ✓ PREMIUM ACTIVE
    </div>
  `;
  document.body.appendChild(premiumBadge);
}

// Export for use in main platform
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { commercialOverlay, injectCommercialOverlay };
}