# ═══════════════════════════════════════════════════════════════════════════
# gemini_sentiment_helper.R
#
# Lightweight Gemini API helper for LLM-augmented sentiment scoring of
# policy text (Project 20). Adapted from the full scoring pipeline of
# Project 06 (score_response_llm), simplified for continuous sentiment
# scoring on a [-1, +1] scale.
#
# Usage:
#   source("files/gemini_sentiment_helper.R")
#   Sys.setenv(GEMINI_API_KEY = "your_key_here")
#   result <- gemini_sentiment("Inflation remains well below our target.")
#
# Author: Francisco Martin-Gomez
# Date: 2026-04-14
# ═══════════════════════════════════════════════════════════════════════════

library(httr2)
library(jsonlite)
library(digest)

# ── Configuration ────────────────────────────────────────────────────────────

GEMINI_MODEL       <- "gemini-3.1-flash-lite-preview"
GEMINI_ENDPOINT    <- "https://generativelanguage.googleapis.com/v1beta/models"
CACHE_DIR          <- "files/cache_gemini_sentiment"
DEFAULT_TEMPERATURE<- 0.1  # low for consistency

if (!dir.exists(CACHE_DIR)) dir.create(CACHE_DIR, recursive = TRUE)


# ── Prompt builder ───────────────────────────────────────────────────────────

build_sentiment_prompt <- function(text) {
  paste0(
    "You are an expert central-bank sentiment analyst specialised in ",
    "monetary policy communications.\n\n",
    "Score the following ECB monetary policy statement on a continuous ",
    "scale from -1.0 to +1.0:\n",
    "  -1.0 = strongly negative / cautious / crisis language\n",
    "   0.0 = neutral / balanced\n",
    "  +1.0 = strongly positive / confident / accommodative language\n\n",
    "Consider: linguistic hedging, forward guidance tone, risk framing, ",
    "and implicit policy stance.\n\n",
    "Return ONLY a JSON object with this exact structure:\n",
    '{"score": <float>, "justification": "<brief 1-sentence explanation>"}\n\n',
    "Statement:\n", text
  )
}


# ── Core API call with caching ───────────────────────────────────────────────

gemini_sentiment <- function(text, temperature = DEFAULT_TEMPERATURE,
                              use_cache = TRUE, verbose = FALSE) {
  
  api_key <- Sys.getenv("GEMINI_API_KEY")
  if (nchar(api_key) == 0) {
    stop("GEMINI_API_KEY environment variable not set. ",
         "Set it with Sys.setenv(GEMINI_API_KEY = 'your_key').")
  }
  
  prompt <- build_sentiment_prompt(text)
  
  # ── Cache check ────────────────────────────────────────────────────────────
  cache_hash <- digest(paste(prompt, GEMINI_MODEL, temperature), algo = "md5")
  cache_file <- file.path(CACHE_DIR, paste0(cache_hash, ".rds"))
  
  if (use_cache && file.exists(cache_file)) {
    if (verbose) cat(sprintf("  [cache hit: %s]\n", substr(cache_hash, 1, 8)))
    return(readRDS(cache_file))
  }
  
  # ── API call ───────────────────────────────────────────────────────────────
  url <- sprintf("%s/%s:generateContent?key=%s",
                 GEMINI_ENDPOINT, GEMINI_MODEL, api_key)
  
  body <- list(
    contents = list(list(parts = list(list(text = prompt)))),
    generationConfig = list(
      temperature      = temperature,
      maxOutputTokens  = 200,
      responseMimeType = "application/json"
    )
  )
  
  result <- tryCatch({
    resp <- request(url) |>
      req_body_json(body) |>
      req_timeout(30) |>
      req_retry(max_tries = 3, backoff = ~ 2 ^ .x) |>
      req_perform()
    
    parsed <- resp |> resp_body_json()
    raw    <- parsed$candidates[[1]]$content$parts[[1]]$text
    
    # ── Parse JSON response ────────────────────────────────────────────────
    parsed_json <- fromJSON(raw)
    
    list(
      score          = as.numeric(parsed_json$score),
      justification  = parsed_json$justification,
      raw_response   = raw,
      status         = "success"
    )
  }, error = function(e) {
    list(score         = NA_real_,
         justification = NA_character_,
         raw_response  = NA_character_,
         status        = paste("error:", e$message))
  })
  
  # ── Cache successful results ───────────────────────────────────────────────
  if (use_cache && result$status == "success") {
    saveRDS(result, cache_file)
  }
  
  if (verbose) {
    cat(sprintf("  score = %.2f | %s\n",
                result$score, substr(result$justification, 1, 60)))
  }
  
  result
}


# ── Batch wrapper ────────────────────────────────────────────────────────────

score_corpus_gemini <- function(corpus, text_col = "text",
                                  sleep_between = 0.5,
                                  verbose = TRUE) {
  
  n <- nrow(corpus)
  if (verbose) cat(sprintf("Scoring %d documents with Gemini...\n", n))
  
  results <- vector("list", n)
  for (i in seq_len(n)) {
    if (verbose && i %% 5 == 0) cat(sprintf("  [%d/%d]\n", i, n))
    results[[i]] <- gemini_sentiment(corpus[[text_col]][i],
                                       verbose = FALSE)
    Sys.sleep(sleep_between)  # rate limiting
  }
  
  corpus |>
    dplyr::mutate(
      llm_score         = sapply(results, function(r) r$score),
      llm_justification = sapply(results, function(r) r$justification),
      llm_status        = sapply(results, function(r) r$status)
    )
}
