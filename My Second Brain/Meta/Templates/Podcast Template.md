---
category:
  - "[[Podcasts]]"
host: 
rating: 
tags:
  - podcast
created: 2023-12-14T11:54
updated: 2023-12-25T10:53
---
## Episodes

```dataview
table without id
	link(file.link, "Ep. " + string(episode)) as Episode,
	guests as Guest,
	rating as Rating
where
	contains(category,[[Podcast episodes]]) and
	contains(show,this.file.link)
sort episode desc
```