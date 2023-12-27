---
category:
  - "[[People]]"
tags:
  - people
birthday: 
org: 
created: 2023-12-14T11:54
updated: 2023-12-24T23:49
---
## Meetings

```dataview
table without id
	file.link as Meeting,
	date as Date
where
	contains(category,[[Meetings]]) and
	contains(people,this.file.link)
sort file.name desc
```