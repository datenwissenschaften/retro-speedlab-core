const PLATFORM_SUFFIX = /-[A-Za-z0-9]+-v\d+$/
const WORD_BOUNDARY = /([a-z])(?=[A-Z0-9])|([0-9])(?=[A-Za-z])|([A-Z])(?=[A-Z][a-z])/g

export const fmt = (value, digits = 0) => value == null ? '—' : Intl.NumberFormat('en', { maximumFractionDigits: digits }).format(value)

export const percent = value => `${Math.round(value * 100)}%`

export const words = name => name.replace(WORD_BOUNDARY, '$1$2$3 ')

export const gameTitle = gameId => words(gameId.replace(PLATFORM_SUFFIX, ''))
  .split(' ')
  .map(word => word === 'N' ? "'n'" : word)
  .join(' ')
