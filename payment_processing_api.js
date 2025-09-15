/**
 * GOMNA AI PAYMENT PROCESSING API
 * Backend API for handling payment transactions, subscriptions, and billing
 */

const express = require('express');
const router = express.Router();

// Payment Gateway Integrations
class PaymentProcessor {
    constructor() {
        this.stripe = null;
        this.paypal = null;
        this.crypto = null;
        this.initializeProviders();
    }

    async initializeProviders() {
        // Initialize Stripe
        if (process.env.STRIPE_SECRET_KEY) {
            this.stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);
        }
        
        // Initialize PayPal
        if (process.env.PAYPAL_CLIENT_ID) {
            // PayPal SDK initialization
        }
        
        // Initialize Crypto Payment Gateway
        if (process.env.COINBASE_API_KEY) {
            // Coinbase Commerce initialization
        }
    }

    /**
     * Process credit/debit card payment
     */
    async processCardPayment(paymentData) {
        try {
            // Create payment intent
            const paymentIntent = await this.stripe.paymentIntents.create({
                amount: paymentData.amount * 100, // Convert to cents
                currency: paymentData.currency || 'usd',
                payment_method_types: ['card'],
                metadata: {
                    userId: paymentData.userId,
                    planId: paymentData.planId,
                    type: 'subscription'
                }
            });

            // Create customer
            const customer = await this.stripe.customers.create({
                email: paymentData.email,
                name: `${paymentData.firstName} ${paymentData.lastName}`,
                metadata: {
                    userId: paymentData.userId
                }
            });

            // Create subscription
            const subscription = await this.stripe.subscriptions.create({
                customer: customer.id,
                items: [{
                    price: await this.getStripePriceId(paymentData.planId)
                }],
                payment_settings: {
                    payment_method_types: ['card'],
                    save_default_payment_method: 'on_subscription'
                },
                expand: ['latest_invoice.payment_intent']
            });

            return {
                success: true,
                paymentIntentId: paymentIntent.id,
                customerId: customer.id,
                subscriptionId: subscription.id,
                clientSecret: paymentIntent.client_secret
            };
        } catch (error) {
            console.error('Card payment error:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Process bank transfer payment
     */
    async processBankTransfer(paymentData) {
        try {
            // Create ACH payment source
            const source = await this.stripe.sources.create({
                type: 'ach_debit',
                currency: 'usd',
                owner: {
                    email: paymentData.email,
                    name: `${paymentData.firstName} ${paymentData.lastName}`
                },
                ach_debit: {
                    account_number: paymentData.accountNumber,
                    routing_number: paymentData.routingNumber,
                    account_holder_type: 'individual'
                }
            });

            // Create charge
            const charge = await this.stripe.charges.create({
                amount: paymentData.amount * 100,
                currency: 'usd',
                source: source.id,
                description: `Gomna AI ${paymentData.planName} Subscription`
            });

            return {
                success: true,
                chargeId: charge.id,
                sourceId: source.id
            };
        } catch (error) {
            console.error('Bank transfer error:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Process cryptocurrency payment
     */
    async processCryptoPayment(paymentData) {
        try {
            // Create crypto charge using Coinbase Commerce
            const charge = {
                name: `Gomna AI ${paymentData.planName} Subscription`,
                description: 'AI Trading Platform Subscription',
                pricing_type: 'fixed_price',
                local_price: {
                    amount: paymentData.amount.toString(),
                    currency: 'USD'
                },
                metadata: {
                    userId: paymentData.userId,
                    planId: paymentData.planId
                },
                redirect_url: `${process.env.BASE_URL}/payment/success`,
                cancel_url: `${process.env.BASE_URL}/payment/cancel`
            };

            // Mock response for demo
            return {
                success: true,
                chargeId: 'CHG_' + Date.now(),
                walletAddress: this.generateCryptoWallet(paymentData.cryptoType),
                amount: paymentData.amount,
                cryptoAmount: this.calculateCryptoAmount(paymentData.amount, paymentData.cryptoType),
                expiresAt: new Date(Date.now() + 30 * 60 * 1000) // 30 minutes
            };
        } catch (error) {
            console.error('Crypto payment error:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Get Stripe price ID for subscription plan
     */
    async getStripePriceId(planId) {
        const priceMap = {
            'starter': process.env.STRIPE_STARTER_PRICE_ID || 'price_starter',
            'professional': process.env.STRIPE_PRO_PRICE_ID || 'price_professional',
            'institutional': process.env.STRIPE_INST_PRICE_ID || 'price_institutional'
        };
        return priceMap[planId] || priceMap['professional'];
    }

    /**
     * Generate crypto wallet address
     */
    generateCryptoWallet(cryptoType) {
        const wallets = {
            'btc': '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',
            'eth': '0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb7',
            'usdt': 'TN3W4H6rK2ce4vX9YnFQHwKENnHjoxb3m9',
            'usdc': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48'
        };
        return wallets[cryptoType] || wallets['btc'];
    }

    /**
     * Calculate crypto amount based on USD
     */
    calculateCryptoAmount(usdAmount, cryptoType) {
        // Mock exchange rates
        const rates = {
            'btc': 0.000023,
            'eth': 0.00042,
            'usdt': 1.0,
            'usdc': 1.0
        };
        return (usdAmount * (rates[cryptoType] || 1)).toFixed(8);
    }
}

// API Routes
const paymentProcessor = new PaymentProcessor();

/**
 * Create new subscription
 */
router.post('/api/payments/subscribe', async (req, res) => {
    try {
        const { paymentMethod, ...paymentData } = req.body;
        
        let result;
        switch (paymentMethod) {
            case 'card':
                result = await paymentProcessor.processCardPayment(paymentData);
                break;
            case 'bank':
                result = await paymentProcessor.processBankTransfer(paymentData);
                break;
            case 'crypto':
                result = await paymentProcessor.processCryptoPayment(paymentData);
                break;
            default:
                throw new Error('Invalid payment method');
        }
        
        if (result.success) {
            // Save transaction to database
            await saveTransaction({
                userId: paymentData.userId,
                amount: paymentData.amount,
                method: paymentMethod,
                status: 'pending',
                metadata: result
            });
            
            res.json({
                success: true,
                data: result
            });
        } else {
            res.status(400).json({
                success: false,
                error: result.error
            });
        }
    } catch (error) {
        console.error('Subscription error:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to process subscription'
        });
    }
});

/**
 * Confirm payment
 */
router.post('/api/payments/confirm', async (req, res) => {
    try {
        const { paymentIntentId, paymentMethodId } = req.body;
        
        const paymentIntent = await paymentProcessor.stripe.paymentIntents.confirm(
            paymentIntentId,
            { payment_method: paymentMethodId }
        );
        
        res.json({
            success: true,
            status: paymentIntent.status
        });
    } catch (error) {
        console.error('Payment confirmation error:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to confirm payment'
        });
    }
});

/**
 * Get subscription status
 */
router.get('/api/payments/subscription/:userId', async (req, res) => {
    try {
        const { userId } = req.params;
        
        // Get user's subscription from database
        const subscription = await getSubscription(userId);
        
        if (subscription && subscription.stripeSubscriptionId) {
            const stripeSubscription = await paymentProcessor.stripe.subscriptions.retrieve(
                subscription.stripeSubscriptionId
            );
            
            res.json({
                success: true,
                subscription: {
                    status: stripeSubscription.status,
                    currentPeriodEnd: stripeSubscription.current_period_end,
                    planId: subscription.planId,
                    features: getSubscriptionFeatures(subscription.planId)
                }
            });
        } else {
            res.json({
                success: true,
                subscription: null
            });
        }
    } catch (error) {
        console.error('Get subscription error:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to retrieve subscription'
        });
    }
});

/**
 * Cancel subscription
 */
router.post('/api/payments/cancel', async (req, res) => {
    try {
        const { userId, subscriptionId } = req.body;
        
        const subscription = await paymentProcessor.stripe.subscriptions.update(
            subscriptionId,
            { cancel_at_period_end: true }
        );
        
        res.json({
            success: true,
            message: 'Subscription will be cancelled at the end of the current period',
            cancelAt: subscription.cancel_at
        });
    } catch (error) {
        console.error('Cancel subscription error:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to cancel subscription'
        });
    }
});

/**
 * Update payment method
 */
router.post('/api/payments/update-method', async (req, res) => {
    try {
        const { customerId, paymentMethodId } = req.body;
        
        await paymentProcessor.stripe.paymentMethods.attach(
            paymentMethodId,
            { customer: customerId }
        );
        
        await paymentProcessor.stripe.customers.update(customerId, {
            invoice_settings: {
                default_payment_method: paymentMethodId
            }
        });
        
        res.json({
            success: true,
            message: 'Payment method updated successfully'
        });
    } catch (error) {
        console.error('Update payment method error:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to update payment method'
        });
    }
});

/**
 * Get payment history
 */
router.get('/api/payments/history/:userId', async (req, res) => {
    try {
        const { userId } = req.params;
        
        // Get payment history from database
        const payments = await getPaymentHistory(userId);
        
        res.json({
            success: true,
            payments: payments
        });
    } catch (error) {
        console.error('Get payment history error:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to retrieve payment history'
        });
    }
});

/**
 * Webhook handler for payment events
 */
router.post('/api/payments/webhook', express.raw({ type: 'application/json' }), async (req, res) => {
    try {
        const sig = req.headers['stripe-signature'];
        let event;
        
        try {
            event = paymentProcessor.stripe.webhooks.constructEvent(
                req.body,
                sig,
                process.env.STRIPE_WEBHOOK_SECRET
            );
        } catch (err) {
            console.error('Webhook signature verification failed:', err);
            return res.status(400).send(`Webhook Error: ${err.message}`);
        }
        
        // Handle the event
        switch (event.type) {
            case 'payment_intent.succeeded':
                await handlePaymentSuccess(event.data.object);
                break;
            case 'payment_intent.payment_failed':
                await handlePaymentFailure(event.data.object);
                break;
            case 'subscription.created':
                await handleSubscriptionCreated(event.data.object);
                break;
            case 'subscription.updated':
                await handleSubscriptionUpdated(event.data.object);
                break;
            case 'subscription.deleted':
                await handleSubscriptionCancelled(event.data.object);
                break;
            case 'invoice.payment_succeeded':
                await handleInvoicePaymentSuccess(event.data.object);
                break;
            default:
                console.log(`Unhandled event type ${event.type}`);
        }
        
        res.json({ received: true });
    } catch (error) {
        console.error('Webhook error:', error);
        res.status(500).json({
            success: false,
            error: 'Webhook processing failed'
        });
    }
});

// Database helper functions (these would connect to your actual database)
async function saveTransaction(transactionData) {
    // Save to database
    console.log('Saving transaction:', transactionData);
    return true;
}

async function getSubscription(userId) {
    // Get from database
    return {
        userId,
        planId: 'professional',
        stripeSubscriptionId: 'sub_test123'
    };
}

async function getPaymentHistory(userId) {
    // Get from database
    return [
        {
            id: 'pay_1',
            amount: 499,
            currency: 'usd',
            status: 'succeeded',
            created: Date.now() - 30 * 24 * 60 * 60 * 1000,
            description: 'Professional Plan - Monthly'
        }
    ];
}

function getSubscriptionFeatures(planId) {
    const features = {
        'starter': [
            'Basic AI Trading Signals',
            'Portfolio Analytics',
            '5 Trading Strategies',
            'Email Support'
        ],
        'professional': [
            'Advanced AI Trading Signals',
            'Real-time Portfolio Optimization',
            '20+ Trading Strategies',
            'Priority Support',
            'API Access'
        ],
        'institutional': [
            'Enterprise AI Trading Suite',
            'Unlimited Trading Strategies',
            'White-glove Support',
            'Custom Strategy Development',
            'Dedicated Account Manager'
        ]
    };
    return features[planId] || features['professional'];
}

// Event handlers
async function handlePaymentSuccess(paymentIntent) {
    console.log('Payment succeeded:', paymentIntent.id);
    // Update database, send confirmation email, etc.
}

async function handlePaymentFailure(paymentIntent) {
    console.log('Payment failed:', paymentIntent.id);
    // Update database, send failure notification, etc.
}

async function handleSubscriptionCreated(subscription) {
    console.log('Subscription created:', subscription.id);
    // Update user's subscription status in database
}

async function handleSubscriptionUpdated(subscription) {
    console.log('Subscription updated:', subscription.id);
    // Update subscription details in database
}

async function handleSubscriptionCancelled(subscription) {
    console.log('Subscription cancelled:', subscription.id);
    // Update user's access, send cancellation email
}

async function handleInvoicePaymentSuccess(invoice) {
    console.log('Invoice payment succeeded:', invoice.id);
    // Update payment records, extend subscription period
}

module.exports = router;