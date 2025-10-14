/**
 * ENTERPRISE AI ASSISTANT FOR TRADING PLATFORM
 * 
 * Advanced AI Assistant powered by state-of-the-art LLMs (OpenAI GPT-4/Anthropic Claude)
 * for providing intelligent trading insights, market analysis, and strategic recommendations.
 * 
 * Features:
 * - Multi-modal data fusion and analysis
 * - Real-time market sentiment processing  
 * - Advanced risk assessment and portfolio optimization
 * - Natural language query interface for complex trading scenarios
 * - Academic-grade backtesting interpretation
 * - Regulatory compliance and industry standard adherence
 */

export interface AIAssistantConfig {
  provider: 'openai' | 'anthropic' | 'hybrid'
  model: string
  temperature: number
  maxTokens: number
  systemPrompt?: string
  enableMemory: boolean
  riskLevel: 'conservative' | 'moderate' | 'aggressive'
  tradingExperience: 'beginner' | 'intermediate' | 'expert'
}

export interface MarketContext {
  currentPrices: Record<string, number>
  marketSentiment: any
  economicIndicators: any
  portfolioState: any
  activeStrategies: any[]
  riskMetrics: any
  recentTrades: any[]
}

export interface AIResponse {
  response: string
  confidence: number
  reasoning: string[]
  actionableInsights: string[]
  riskWarnings: string[]
  followUpQuestions: string[]
  citations: string[]
  timestamp: string
  processingTime: number
}

export interface ConversationContext {
  conversationId: string
  messages: Array<{
    role: 'user' | 'assistant' | 'system'
    content: string
    timestamp: string
    metadata?: any
  }>
  context: MarketContext
  userPreferences: {
    riskTolerance: number
    tradingGoals: string[]
    timeHorizon: string
    experienceLevel: string
  }
}

export class IntelligentTradingAssistant {
  private config: AIAssistantConfig
  private conversations: Map<string, ConversationContext> = new Map()
  private apiKey: string
  private baseUrl: string
  
  constructor(config: AIAssistantConfig, apiKey: string) {
    this.config = {
      provider: 'openai',
      model: 'gpt-4o',
      temperature: 0.1, // Low temperature for consistent, reliable responses
      maxTokens: 2000,
      enableMemory: true,
      riskLevel: 'moderate',
      tradingExperience: 'intermediate',
      ...config
    }
    this.apiKey = apiKey
    this.baseUrl = this.getBaseUrl()
    
    console.log('🤖 Intelligent Trading Assistant initialized with:', {
      provider: this.config.provider,
      model: this.config.model,
      riskLevel: this.config.riskLevel
    })
  }
  
  private getBaseUrl(): string {
    switch (this.config.provider) {
      case 'openai':
        return 'https://api.openai.com/v1'
      case 'anthropic':
        return 'https://api.anthropic.com/v1'
      default:
        return 'https://api.openai.com/v1'
    }
  }
  
  /**
   * Generate comprehensive system prompt for trading context
   */
  private generateSystemPrompt(context: MarketContext): string {
    const riskGuidance = {
      conservative: "Prioritize capital preservation and low-risk strategies. Recommend conservative position sizing and thorough risk management.",
      moderate: "Balance risk and return. Suggest diversified approaches with moderate position sizing and standard risk controls.",
      aggressive: "Focus on growth opportunities while maintaining prudent risk management. Allow for higher position sizes with appropriate safeguards."
    }
    
    const experienceGuidance = {
      beginner: "Explain concepts clearly and provide educational context. Focus on fundamentals and simple strategies.",
      intermediate: "Provide balanced analysis with some technical depth. Include intermediate-level strategies and risk concepts.",
      expert: "Offer sophisticated analysis with advanced concepts. Assume familiarity with complex strategies and quantitative metrics."
    }
    
    return `You are an elite AI Trading Assistant for the GOMNA Enterprise AI Trading System, serving sophisticated investors in arbitrage trading strategies.

CORE IDENTITY & EXPERTISE:
- Senior Quantitative Analyst with expertise in arbitrage, algorithmic trading, and risk management
- Specialized in multi-modal data fusion, hyperbolic space analysis, and AI-enhanced trading strategies  
- Academic rigor combined with practical industry experience
- Regulatory compliance expert (SEC, CFTC, MiFID II)

CURRENT MARKET CONTEXT:
- Portfolio Value: $${context.portfolioState?.totalValue?.toLocaleString() || 'N/A'}
- Active Positions: ${Object.keys(context.portfolioState?.assets || {}).join(', ') || 'None'}
- Market Sentiment: ${context.marketSentiment?.fearGreedIndex || 'N/A'}/100 (Fear & Greed Index)
- Recent Performance: ${context.portfolioState?.monthlyChange || 'N/A'}% monthly change
- Risk Level: ${this.config.riskLevel.toUpperCase()}

RISK MANAGEMENT PHILOSOPHY:
${riskGuidance[this.config.riskLevel]}

COMMUNICATION STYLE:
${experienceGuidance[this.config.tradingExperience]}

KEY RESPONSIBILITIES:
1. Analyze arbitrage opportunities with statistical rigor
2. Provide real-time market insights and sentiment analysis
3. Interpret backtesting results with academic standards
4. Assess portfolio risk using advanced metrics (VaR, CVaR, Sharpe, etc.)
5. Recommend strategic adjustments based on market conditions
6. Explain complex trading concepts in accessible terms
7. Identify potential regulatory or compliance considerations

ANALYTICAL FRAMEWORK:
- Use quantitative evidence and statistical significance
- Consider market microstructure and execution costs
- Incorporate behavioral finance and market psychology
- Apply modern portfolio theory and risk parity concepts
- Reference academic literature and industry best practices

RESPONSE GUIDELINES:
- Always lead with key insights in bullet points
- Provide confidence levels for recommendations (0-100%)
- Include specific risk warnings and position sizing guidance
- Offer actionable next steps with clear timelines
- Cite relevant data sources and reasoning
- Maintain professional, authoritative tone while being accessible

CURRENT SYSTEM CAPABILITIES:
- 100% reliable backtesting engine with bias correction
- Autonomous AI agent with smart decision-making
- Monte Carlo simulation and stress testing
- Hyperbolic space pattern recognition
- Cross-asset arbitrage opportunity identification
- Real-time sentiment and economic indicator analysis

Remember: You serve sophisticated investors who value precision, evidence-based analysis, and actionable insights. Every recommendation should be backed by data and aligned with their risk profile and investment objectives.`
  }
  
  /**
   * Process user query with full market context awareness
   */
  async processQuery(
    query: string, 
    context: MarketContext, 
    conversationId?: string
  ): Promise<AIResponse> {
    const startTime = Date.now()
    
    try {
      // Create or retrieve conversation context
      const convId = conversationId || this.generateConversationId()
      let conversation = this.conversations.get(convId)
      
      if (!conversation) {
        conversation = {
          conversationId: convId,
          messages: [],
          context,
          userPreferences: {
            riskTolerance: this.mapRiskLevel(this.config.riskLevel),
            tradingGoals: ['capital growth', 'risk management'],
            timeHorizon: 'medium-term',
            experienceLevel: this.config.tradingExperience
          }
        }
        this.conversations.set(convId, conversation)
      }
      
      // Update context with latest market data
      conversation.context = { ...conversation.context, ...context }
      
      // Add user message to conversation
      conversation.messages.push({
        role: 'user',
        content: query,
        timestamp: new Date().toISOString()
      })
      
      // Generate enhanced system prompt with current context
      const systemPrompt = this.generateSystemPrompt(context)
      
      // Prepare messages for API call
      const messages = [
        { role: 'system', content: systemPrompt },
        ...conversation.messages.slice(-10) // Keep last 10 messages for context
      ]
      
      // Call AI API
      const aiResponse = await this.callAIAPI(messages)
      
      // Add AI response to conversation
      conversation.messages.push({
        role: 'assistant',
        content: aiResponse.response,
        timestamp: new Date().toISOString(),
        metadata: { confidence: aiResponse.confidence }
      })
      
      // Process and enhance the response
      const enhancedResponse = await this.enhanceResponse(aiResponse, query, context)
      
      const processingTime = Date.now() - startTime
      
      return {
        ...enhancedResponse,
        processingTime,
        timestamp: new Date().toISOString()
      }
      
    } catch (error) {
      console.error('❌ AI Assistant query processing failed:', error)
      
      return {
        response: "I apologize, but I'm experiencing technical difficulties. Please try again in a moment, or contact support if the issue persists.",
        confidence: 0,
        reasoning: ['Technical error occurred during processing'],
        actionableInsights: ['Retry the query', 'Check system status', 'Contact support if needed'],
        riskWarnings: ['AI Assistant temporarily unavailable'],
        followUpQuestions: [],
        citations: [],
        timestamp: new Date().toISOString(),
        processingTime: Date.now() - startTime
      }
    }
  }
  
  /**
   * Call the appropriate AI API based on configuration
   */
  private async callAIAPI(messages: any[]): Promise<{ response: string, confidence: number }> {
    const headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${this.apiKey}`
    }
    
    let requestBody: any
    let endpoint: string
    
    if (this.config.provider === 'openai') {
      endpoint = `${this.baseUrl}/chat/completions`
      requestBody = {
        model: this.config.model,
        messages,
        temperature: this.config.temperature,
        max_tokens: this.config.maxTokens,
        response_format: { type: "text" }
      }
    } else if (this.config.provider === 'anthropic') {
      endpoint = `${this.baseUrl}/messages`
      headers['anthropic-version'] = '2023-06-01'
      requestBody = {
        model: this.config.model,
        max_tokens: this.config.maxTokens,
        temperature: this.config.temperature,
        messages: messages.filter(m => m.role !== 'system'),
        system: messages.find(m => m.role === 'system')?.content || ''
      }
    } else {
      throw new Error('Unsupported AI provider')
    }
    
    const response = await fetch(endpoint, {
      method: 'POST',
      headers,
      body: JSON.stringify(requestBody)
    })
    
    if (!response.ok) {
      throw new Error(`AI API request failed: ${response.status} ${response.statusText}`)
    }
    
    const data = await response.json()
    
    let content: string
    if (this.config.provider === 'openai') {
      content = data.choices[0]?.message?.content || ''
    } else if (this.config.provider === 'anthropic') {
      content = data.content[0]?.text || ''
    } else {
      content = ''
    }
    
    // Calculate confidence based on response length and coherence
    const confidence = this.calculateResponseConfidence(content)
    
    return { response: content, confidence }
  }
  
  /**
   * Enhance AI response with structured insights and warnings
   */
  private async enhanceResponse(
    aiResponse: { response: string, confidence: number },
    originalQuery: string,
    context: MarketContext
  ): Promise<AIResponse> {
    // Extract structured information from response
    const reasoning = this.extractReasoning(aiResponse.response)
    const actionableInsights = this.extractActionableInsights(aiResponse.response)
    const riskWarnings = this.extractRiskWarnings(aiResponse.response, context)
    const followUpQuestions = this.generateFollowUpQuestions(originalQuery, aiResponse.response)
    const citations = this.extractCitations(aiResponse.response)
    
    return {
      response: aiResponse.response,
      confidence: Math.min(aiResponse.confidence, 95), // Cap at 95% for prudence
      reasoning,
      actionableInsights,
      riskWarnings,
      followUpQuestions,
      citations,
      timestamp: new Date().toISOString(),
      processingTime: 0 // Will be set by caller
    }
  }
  
  /**
   * Calculate response confidence based on multiple factors
   */
  private calculateResponseConfidence(content: string): number {
    let confidence = 85 // Base confidence
    
    // Adjust based on response length (too short or too long reduces confidence)
    if (content.length < 100) confidence -= 15
    else if (content.length > 2000) confidence -= 5
    else if (content.length >= 200 && content.length <= 800) confidence += 5
    
    // Look for uncertainty indicators
    const uncertaintyWords = ['maybe', 'might', 'could be', 'possibly', 'unsure', 'unclear']
    const uncertaintyCount = uncertaintyWords.reduce((count, word) => {
      return count + (content.toLowerCase().split(word).length - 1)
    }, 0)
    confidence -= Math.min(uncertaintyCount * 3, 15)
    
    // Look for confidence indicators
    const confidenceWords = ['definitely', 'clearly', 'certainly', 'strong evidence', 'data shows']
    const confidenceCount = confidenceWords.reduce((count, word) => {
      return count + (content.toLowerCase().split(word).length - 1)
    }, 0)
    confidence += Math.min(confidenceCount * 2, 10)
    
    return Math.max(10, Math.min(95, confidence))
  }
  
  /**
   * Extract reasoning points from AI response
   */
  private extractReasoning(response: string): string[] {
    const reasoning: string[] = []
    
    // Look for bullet points or numbered lists
    const bulletPoints = response.match(/[•\-\*]\s*(.+)/g) || []
    bulletPoints.forEach(point => {
      const cleaned = point.replace(/^[•\-\*]\s*/, '').trim()
      if (cleaned.length > 10) reasoning.push(cleaned)
    })
    
    // Look for "because" or "due to" statements
    const becauseStatements = response.match(/because\s+([^.]+)/gi) || []
    becauseStatements.forEach(statement => {
      const cleaned = statement.replace(/^because\s+/i, '').trim()
      if (cleaned.length > 10) reasoning.push(cleaned)
    })
    
    // If no structured reasoning found, extract key sentences
    if (reasoning.length === 0) {
      const sentences = response.split(/[.!?]+/)
      sentences.slice(1, 4).forEach(sentence => {
        const cleaned = sentence.trim()
        if (cleaned.length > 20) reasoning.push(cleaned)
      })
    }
    
    return reasoning.slice(0, 5) // Limit to 5 reasoning points
  }
  
  /**
   * Extract actionable insights from AI response
   */
  private extractActionableInsights(response: string): string[] {
    const insights: string[] = []
    
    // Look for action verbs and recommendations
    const actionPatterns = [
      /consider\s+([^.]+)/gi,
      /recommend\s+([^.]+)/gi,
      /suggest\s+([^.]+)/gi,
      /should\s+([^.]+)/gi,
      /next step[s]?:\s*([^.]+)/gi
    ]
    
    actionPatterns.forEach(pattern => {
      const matches = response.match(pattern) || []
      matches.forEach(match => {
        const cleaned = match.replace(pattern, '$1').trim()
        if (cleaned.length > 10) insights.push(cleaned)
      })
    })
    
    return insights.slice(0, 5) // Limit to 5 insights
  }
  
  /**
   * Extract and generate risk warnings
   */
  private extractRiskWarnings(response: string, context: MarketContext): string[] {
    const warnings: string[] = []
    
    // Extract explicit warnings from response
    const warningPatterns = [
      /warning[s]?:\s*([^.]+)/gi,
      /caution[s]?:\s*([^.]+)/gi,
      /risk[s]?:\s*([^.]+)/gi,
      /be careful\s+([^.]+)/gi
    ]
    
    warningPatterns.forEach(pattern => {
      const matches = response.match(pattern) || []
      matches.forEach(match => {
        const cleaned = match.replace(pattern, '$1').trim()
        if (cleaned.length > 10) warnings.push(cleaned)
      })
    })
    
    // Add context-based warnings
    if (context.portfolioState?.totalValue) {
      const drawdown = context.portfolioState.monthlyChange
      if (drawdown < -5) {
        warnings.push(`Portfolio experiencing ${Math.abs(drawdown)}% drawdown - consider reducing position sizes`)
      }
    }
    
    // Add market condition warnings
    if (context.marketSentiment?.fearGreedIndex) {
      if (context.marketSentiment.fearGreedIndex > 80) {
        warnings.push('Extreme greed detected - markets may be overextended, exercise caution')
      } else if (context.marketSentiment.fearGreedIndex < 20) {
        warnings.push('Extreme fear detected - potential opportunity but high volatility expected')
      }
    }
    
    return warnings.slice(0, 5) // Limit to 5 warnings
  }
  
  /**
   * Generate relevant follow-up questions
   */
  private generateFollowUpQuestions(originalQuery: string, response: string): string[] {
    const questions: string[] = []
    
    // Query-specific follow-ups
    if (originalQuery.toLowerCase().includes('risk')) {
      questions.push("Would you like me to run a Monte Carlo simulation to quantify this risk?")
      questions.push("Should we analyze the correlation impact on your portfolio?")
    }
    
    if (originalQuery.toLowerCase().includes('strategy')) {
      questions.push("Would you like to see the backtesting results for this strategy?")
      questions.push("Should I compare this with alternative strategies?")
    }
    
    if (originalQuery.toLowerCase().includes('market') || originalQuery.toLowerCase().includes('price')) {
      questions.push("Would you like current arbitrage opportunities in this market?")
      questions.push("Should I analyze the sentiment indicators for this asset?")
    }
    
    // General follow-ups
    questions.push("Would you like me to explain any specific aspect in more detail?")
    questions.push("Should I analyze how this affects your current portfolio allocation?")
    
    return questions.slice(0, 3) // Limit to 3 questions
  }
  
  /**
   * Extract citations and data sources
   */
  private extractCitations(response: string): string[] {
    const citations: string[] = []
    
    // Look for mentions of data sources, metrics, or references
    const sourcePatterns = [
      /according to\s+([^,\.]+)/gi,
      /based on\s+([^,\.]+)/gi,
      /studies show/gi,
      /research indicates/gi,
      /data from\s+([^,\.]+)/gi
    ]
    
    sourcePatterns.forEach(pattern => {
      const matches = response.match(pattern) || []
      matches.forEach(match => {
        citations.push(match.trim())
      })
    })
    
    // Add default data sources
    citations.push('GOMNA Real-time Market Data')
    citations.push('Enterprise Backtesting Engine')
    citations.push('Multi-Modal Sentiment Analysis')
    
    return Array.from(new Set(citations)).slice(0, 5) // Remove duplicates and limit
  }
  
  /**
   * Utility methods
   */
  private generateConversationId(): string {
    return `conv_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`
  }
  
  private mapRiskLevel(level: string): number {
    const mapping = { conservative: 25, moderate: 50, aggressive: 75 }
    return mapping[level as keyof typeof mapping] || 50
  }
  
  /**
   * Get conversation history
   */
  getConversationHistory(conversationId: string): ConversationContext | null {
    return this.conversations.get(conversationId) || null
  }
  
  /**
   * Clear conversation history
   */
  clearConversationHistory(conversationId: string): boolean {
    return this.conversations.delete(conversationId)
  }
  
  /**
   * Get assistant status and configuration
   */
  getAssistantStatus(): any {
    return {
      provider: this.config.provider,
      model: this.config.model,
      riskLevel: this.config.riskLevel,
      tradingExperience: this.config.tradingExperience,
      activeConversations: this.conversations.size,
      memoryEnabled: this.config.enableMemory,
      status: 'operational'
    }
  }
}

export default IntelligentTradingAssistant