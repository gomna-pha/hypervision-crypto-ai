// GOMNA AI Tab Testing Script
// Run this in browser console to test tab functionality

console.log('🧪 Starting Tab Functionality Test...');

function testTabSwitching() {
    const tabs = ['portfolio', 'markets', 'transparency', 'investor-portal'];
    let testResults = [];
    
    tabs.forEach((tabName, index) => {
        setTimeout(() => {
            console.log(`\n🔄 Testing ${tabName} tab...`);
            
            // Switch to the tab
            if (typeof switchTab === 'function') {
                switchTab(tabName);
                
                // Check if tab is visible after switch
                setTimeout(() => {
                    const tabElement = document.getElementById(tabName + '-tab');
                    if (tabElement) {
                        const isVisible = !tabElement.classList.contains('hidden') && 
                                        tabElement.classList.contains('active');
                        const hasContent = tabElement.children.length > 0;
                        const hasVisibleHeight = tabElement.offsetHeight > 0;
                        const computedStyle = window.getComputedStyle(tabElement);
                        
                        const result = {
                            tab: tabName,
                            isVisible,
                            hasContent,
                            hasVisibleHeight,
                            childrenCount: tabElement.children.length,
                            display: computedStyle.display,
                            visibility: computedStyle.visibility,
                            opacity: computedStyle.opacity,
                            height: tabElement.offsetHeight
                        };
                        
                        testResults.push(result);
                        
                        console.log(`✅ ${tabName} tab test result:`, result);
                        
                        if (!isVisible || !hasVisibleHeight) {
                            console.error(`❌ ${tabName} tab is not properly visible!`);
                        } else {
                            console.log(`✅ ${tabName} tab is working correctly!`);
                        }
                        
                        // If this is the last test, show summary
                        if (testResults.length === tabs.length) {
                            console.log('\n📊 Test Summary:');
                            testResults.forEach(r => {
                                const status = (r.isVisible && r.hasVisibleHeight) ? '✅' : '❌';
                                console.log(`${status} ${r.tab}: Visible=${r.isVisible}, Height=${r.height}px, Children=${r.childrenCount}`);
                            });
                        }
                    } else {
                        console.error(`❌ Tab element not found: ${tabName}-tab`);
                    }
                }, 500);
                
            } else {
                console.error('❌ switchTab function not found!');
            }
        }, index * 2000); // Stagger tests by 2 seconds each
    });
}

// Start the test
testTabSwitching();

// Also provide manual test functions
window.testPortfolio = () => switchTab('portfolio');
window.testMarkets = () => switchTab('markets');
window.testTransparency = () => switchTab('transparency');
window.testInvestorPortal = () => switchTab('investor-portal');

console.log('🔧 Manual test functions available:');
console.log('- testPortfolio()');
console.log('- testMarkets()');
console.log('- testTransparency()');
console.log('- testInvestorPortal()');